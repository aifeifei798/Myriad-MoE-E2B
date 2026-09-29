from collections import Counter
import copy
import json
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ----------------------------------------------------------------------
# 1. 对抗数据集加载 (复用 50/50 文理对比语料)
# ----------------------------------------------------------------------
class ScalpelDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        messages = [{
            "role": "user",
            "content": item["prompt"]
        }, {
            "role": "assistant",
            "content": item["response"]
        }]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = self.tokenizer(text,
                                max_length=self.max_length,
                                truncation=True,
                                padding="max_length",
                                return_tensors="pt")

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        big_target = int(item["big_target"])
        little_group = int(item["little_group"])

        # 32 微专家映射逻辑：
        # - 文科 (big_target == 0): 映射到 0 ~ 15
        # - 理科 (big_target == 1): 若组号在 0~15 则偏移 +16 映射到 16 ~ 31；若本来就是全局编号则保留
        if big_target == 1 and little_group < 16:
            little_target = little_group + 16
        else:
            little_target = little_group

        # 安全防越界保护（保证索引在 0~31 之内）
        little_target = little_target % 32

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "big_target": torch.tensor(big_target, dtype=torch.long),
            "little_target": torch.tensor(little_target, dtype=torch.long)
        }


# ----------------------------------------------------------------------
# 2. 96 KB 微专家 (Rank=16, Hidden=1536)
# ----------------------------------------------------------------------
class GemmaLoRAMicroExpert(nn.Module):

    def __init__(self,
                 hidden_dim=1536,
                 rank=16,
                 lora_alpha=16.0,
                 device="cuda:0",
                 dtype=torch.bfloat16):
        super().__init__()
        self.scaling = lora_alpha / rank
        self.lora_A = nn.Linear(hidden_dim,
                                rank,
                                bias=False,
                                device=device,
                                dtype=dtype)
        self.lora_B = nn.Linear(rank,
                                hidden_dim,
                                bias=False,
                                device=device,
                                dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling


# ----------------------------------------------------------------------
# 3. Gemma 4 双大核包装器
# ----------------------------------------------------------------------
class ScalpelDualBigWrapper(nn.Module):

    def __init__(self,
                 original_mlp,
                 hidden_dim=1536,
                 rank=16,
                 num_experts=32,
                 device="cuda:0",
                 dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.num_experts = num_experts

        # 文科大核：继承原版 GeGLU MLP，锁死
        self.big_arts = original_mlp
        for p in self.big_arts.parameters():
            p.requires_grad = False

        # 理科大核：1:1 克隆，放开梯度训练
        self.big_sci = copy.deepcopy(original_mlp).to(device)
        for p in self.big_sci.parameters():
            p.requires_grad = True

        # 文理大核路由器 (1536 -> 2)
        self.router_big = nn.Linear(hidden_dim,
                                    2,
                                    bias=False,
                                    device=device,
                                    dtype=dtype)
        self.last_router_big_logits = None

        # 32 微专家池 + 调度器 (1536 -> 32)
        self.router_little = nn.Linear(hidden_dim,
                                       num_experts,
                                       bias=False,
                                       device=device,
                                       dtype=dtype)
        self.last_router_little_logits = None

        self.lora_pool = nn.ModuleList([
            GemmaLoRAMicroExpert(hidden_dim=hidden_dim,
                                 rank=rank,
                                 device=device,
                                 dtype=dtype) for _ in range(num_experts)
        ])
        self.current_little_target = None

    def forward(self, x):
        router_big_logits = self.router_big(x)
        self.last_router_big_logits = router_big_logits
        weights_big = torch.softmax(router_big_logits, dim=-1)

        with torch.no_grad():
            arts_out = self.big_arts(x)
        sci_out = self.big_sci(x)
        big_out = (weights_big[..., 0:1] * arts_out) + (weights_big[..., 1:2] *
                                                        sci_out)

        router_little_logits = self.router_little(x)
        self.last_router_little_logits = router_little_logits

        # 根据班长目标定点训练小核
        batch_size = x.size(0)
        lora_outs = []
        for b in range(batch_size):
            exp_idx = self.current_little_target[b].item()
            lora_outs.append(self.lora_pool[exp_idx](x[b:b + 1]))
        lora_out = torch.cat(lora_outs, dim=0)

        return big_out + 0.3 * lora_out


# ----------------------------------------------------------------------
# 4. 辅助函数：自动定位 Gemma 4 解码层
# ----------------------------------------------------------------------
def locate_gemma_layers(model):
    # 1. Gemma 4 ForConditionalGeneration 结构: model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            return lm.layers
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers

    # 2. 常规多模态封装: model.language_model.layers
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "layers"):
            return lm.layers
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers

    # 3. 纯文本 Gemma / LLaMA 结构: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    # 4. 终极自适应兜底：递归扫描 ModuleList 中包含 35 层的解码器层
    for name, module in model.named_modules():
        if name.endswith("language_model.layers") or name.endswith(
                "model.layers"):
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                return module

    raise RuntimeError("无法定位 Gemma 4 解码器层结构，请检查模型架构属性！")


# ----------------------------------------------------------------------
# 5. 主训练流水线
# ----------------------------------------------------------------------
def main():
    model_id = "aifeifei798/Heretic-Scalpel-E2B"
    data_path = "dual_contrast_data.jsonl"

    assert os.path.exists(
        data_path
    ), f"找不到训练数据 {data_path}！请确保上一个项目的 dual_contrast_data.jsonl 存在或运行 prepare_dual_data.py 生成。"

    print("=" * 70)
    print("🗡️ 启动【Heretic-Scalpel-E2B 手术刀】双大核+微专家改装流水线")
    print("=" * 70)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] 正在载入 2B 手术刀原生模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    # 1. 明确禁用 KV Cache 并启用 input_require_grads (梯度检查点防爆必备)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # 2. 卸载多模态组件（纯文本微调无需占用显存，移至 CPU 立省 3~5 GB）
    for attr in ["vision_tower", "audio_tower", "visual", "audio", "mm_projector"]:
        for parent in [model, getattr(model, "model", None)]:
            if parent is not None and hasattr(parent, attr):
                module = getattr(parent, attr)
                if module is not None:
                    print(f"[*] 将未使用的多模态组件 {attr} 卸载至 CPU...")
                    module.to("cpu")
    torch.cuda.empty_cache()

    # 冻结基础骨架
    for param in model.parameters():
        param.requires_grad = False

    layers = locate_gemma_layers(model)
    num_layers = len(layers)
    hidden_dim = 1536

    print(
        f"[+] 成功捕获 Gemma 4 解码器主干：共 {num_layers} 层 (Hidden Dim: {hidden_dim})")
    print(f"[*] 正在为 35 层全部移植【理科克隆大核】与 32 个 96KB 微专家 (共 1,120 个专家)...")

    for layer in layers:
        layer.mlp = ScalpelDualBigWrapper(layer.mlp,
                                          hidden_dim=hidden_dim,
                                          rank=16,
                                          num_experts=32,
                                          device="cuda:0",
                                          dtype=dtype)

    # 启用梯度检查点防爆
    model.gradient_checkpointing_enable()

    # 参数隔离
    big_sci_params = []
    router_params = []
    lora_params = []

    for layer in layers:
        big_sci_params.extend(
            [p for p in layer.mlp.big_sci.parameters() if p.requires_grad])
        router_params.extend(
            [p for p in layer.mlp.router_big.parameters() if p.requires_grad])
        router_params.extend([
            p for p in layer.mlp.router_little.parameters() if p.requires_grad
        ])
        lora_params.extend(
            [p for p in layer.mlp.lora_pool.parameters() if p.requires_grad])

    print(f"\n[✔] 器官移植完成！参数统计：")
    print(f"    - 理科大核参数量: {sum(p.numel() for p in big_sci_params)/1e6:.1f} M")
    print(
        f"    - 1,120 微专家参数: {sum(p.numel() for p in lora_params)/1e6:.1f} M")
    print(
        f"    - 总微调参数量: 约 {sum(p.numel() for p in (big_sci_params + router_params + lora_params))/1e6:.1f} M (~1.5 GB)"
    )

    # 3. 调小 Micro Batch，等效批次依然是 1 * 16 = 16
    MICRO_BATCH = 1
    GRAD_ACCUM = 16  # 等效 Batch = 16
    MAX_LENGTH = 512

    dataset = ScalpelDataset(data_path, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # 4. 使用 8-bit AdamW，优化器显存占用从 13GB 骤降至 3.2GB
    try:
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW8bit(
            [
                {"params": big_sci_params, "lr": 2e-5, "weight_decay": 0.01},
                {"params": router_params, "lr": 3e-4, "weight_decay": 0.01},
                {"params": lora_params, "lr": 5e-4, "weight_decay": 0.01},
            ]
        )
        print("[✔] 成功启用 bitsandbytes PagedAdamW8bit（释放约 10GB 显存）！")
    except ImportError:
        print("[!] 未检测到 bitsandbytes，使用 PyTorch 原生 AdamW(fused=True)")
        optimizer = torch.optim.AdamW(
            [
                {"params": big_sci_params, "lr": 2e-5, "weight_decay": 0.01},
                {"params": router_params, "lr": 3e-4, "weight_decay": 0.01},
                {"params": lora_params, "lr": 5e-4, "weight_decay": 0.01},
            ],
            fused=True,
        )
    criterion_ce = nn.CrossEntropyLoss()

    total_steps = len(dataloader) // GRAD_ACCUM
    print(f"\n[+] 开始慢火微调特训 (总步数: {total_steps}，预计耗时 10~15 分钟)...")
    model.train()

    start_time = time.time()
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to("cuda:0", non_blocking=True)
        attention_mask = batch["attention_mask"].to("cuda:0",
                                                    non_blocking=True)
        labels = batch["labels"].to("cuda:0", non_blocking=True)
        big_target = batch["big_target"].to("cuda:0", non_blocking=True)
        little_target = batch["little_target"].to("cuda:0", non_blocking=True)

        for layer in layers:
            layer.mlp.current_little_target = little_target

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        lm_loss = outputs.loss

        # 调度器全序列掩码监督
        mask = (attention_mask == 1)
        target_big_seq = big_target.unsqueeze(1).expand(-1, MAX_LENGTH)[mask]
        target_little_seq = little_target.unsqueeze(1).expand(-1,
                                                              MAX_LENGTH)[mask]

        big_router_loss = 0.0
        little_router_loss = 0.0

        for layer in layers:
            active_logits_big = layer.mlp.last_router_big_logits[mask]
            active_logits_little = layer.mlp.last_router_little_logits[mask]

            big_router_loss += criterion_ce(active_logits_big, target_big_seq)
            little_router_loss += criterion_ce(active_logits_little,
                                               target_little_seq)

        total_loss = (lm_loss + 0.1 * (big_router_loss / num_layers) + 0.1 *
                      (little_router_loss / num_layers)) / GRAD_ACCUM
        total_loss.backward()

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

            global_step = (step + 1) // GRAD_ACCUM
            if global_step % 25 == 0 or global_step == total_steps:
                elapsed = time.time() - start_time
                current_loss = total_loss.item() * GRAD_ACCUM
                speed = ((step + 1) * MICRO_BATCH) / elapsed
                print(
                    f"    [Step {global_step:03d}/{total_steps}] Loss: {current_loss:.4f} | LM: {lm_loss.item():.4f} | 速度: {speed:.1f} samples/s | 耗时: {elapsed:.1f}s"
                )

    print(f"\n[✔] 手术刀双大核特训完成！总耗时: {(time.time() - start_time)/60:.2f} 分钟")

    save_path = "scalpel_dualbig_weights.pt"
    print(f"[*] 正在保存手术刀权重至 {save_path}...")

    state_to_save = {}
    for i, layer in enumerate(layers):
        state_to_save[f"layer_{i}_big_sci"] = layer.mlp.big_sci.state_dict()
        state_to_save[
            f"layer_{i}_router_big"] = layer.mlp.router_big.state_dict()
        state_to_save[
            f"layer_{i}_router_little"] = layer.mlp.router_little.state_dict()
        state_to_save[f"layer_{i}_loras"] = layer.mlp.lora_pool.state_dict()

    torch.save(state_to_save, save_path)
    print(f"[✔] 权重成功保存为 {save_path}！")


if __name__ == "__main__":
    main()
