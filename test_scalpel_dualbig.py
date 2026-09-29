from collections import Counter
import copy
import os
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------
# 32 专家语义标签字典 (0~15 文科组, 16~31 理科组)
# ----------------------------------------------------------------------
EXPERT_NAMES = {
    # 0~15: Arts & Humanities Specialists
    0: "Arts_Prose",
    1: "Arts_Poetry",
    2: "Arts_Fiction",
    3: "Arts_Drama",
    4: "Arts_Essay",
    5: "Arts_History",
    6: "Arts_Culture",
    7: "Arts_Linguistics",
    8: "Arts_Philosophy",
    9: "Arts_Ethics",
    10: "Arts_Rhetoric",
    11: "Arts_Translation",
    12: "Arts_Critique",
    13: "Arts_Dialogue",
    14: "Arts_Summary",
    15: "Arts_Chat",
    # 16~31: STEM & Logic Specialists
    16: "Code_Algo",
    17: "Code_DS",
    18: "Code_Debug",
    19: "Code_Arch",
    20: "Code_Syntax",
    21: "Code_Optim",
    22: "Math_Algebra",
    23: "Math_Geo",
    24: "Math_Prob",
    25: "Math_Arith",
    26: "Math_Calculus",
    27: "Math_Logic",
    28: "Sci_Physics",
    29: "Sci_Chem",
    30: "Sci_Biology",
    31: "Sci_General",
}


# ----------------------------------------------------------------------
# 全局遥测监控探针 (跨 35 层全局统计)
# ----------------------------------------------------------------------
class TelemetryMonitor:

    def __init__(self):
        self.reset()

    def reset(self):
        self.arts_weights_sum = 0.0
        self.sci_weights_sum = 0.0
        self.total_tokens_routed = 0
        self.expert_activation_counter = Counter()

    def record_step(self, weights_big, chosen_experts):
        # 记录双大核动态加权
        self.arts_weights_sum += weights_big[..., 0].sum().item()
        self.sci_weights_sum += weights_big[..., 1].sum().item()
        self.total_tokens_routed += weights_big.shape[0] * weights_big.shape[1]

        # 统计微专家激活次数
        flat_experts = chosen_experts.view(-1).tolist()
        self.expert_activation_counter.update(flat_experts)


GLOBAL_TELEMETRY = TelemetryMonitor()


# ----------------------------------------------------------------------
# 1. 96 KB 微专家
# ----------------------------------------------------------------------
class GemmaLoRAMicroExpert(nn.Module):

    def __init__(
        self,
        hidden_dim=1536,
        rank=16,
        lora_alpha=16.0,
        device="cuda:0",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.scaling = lora_alpha / rank
        self.lora_A = nn.Linear(
            hidden_dim, rank, bias=False, device=device, dtype=dtype
        )
        self.lora_B = nn.Linear(
            rank, hidden_dim, bias=False, device=device, dtype=dtype
        )

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling


# ----------------------------------------------------------------------
# 2. 推理专用双大核包装器 (带全息遥测采集)
# ----------------------------------------------------------------------
class ScalpelDualBigInferenceWrapper(nn.Module):

    def __init__(
        self,
        original_mlp,
        hidden_dim=1536,
        rank=16,
        num_experts=32,
        device="cuda:0",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.num_experts = num_experts

        self.big_arts = original_mlp
        self.big_sci = copy.deepcopy(original_mlp).to(device)

        self.router_big = nn.Linear(
            hidden_dim, 2, bias=False, device=device, dtype=dtype
        )
        self.router_little = nn.Linear(
            hidden_dim, num_experts, bias=False, device=device, dtype=dtype
        )

        self.lora_pool = nn.ModuleList([
            GemmaLoRAMicroExpert(
                hidden_dim=hidden_dim,
                rank=rank,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # 1. 双大核软路由
        router_big_logits = self.router_big(x)
        weights_big = torch.softmax(router_big_logits, dim=-1)

        arts_out = self.big_arts(x)
        sci_out = self.big_sci(x)
        big_out = (weights_big[..., 0:1] * arts_out) + (
            weights_big[..., 1:2] * sci_out
        )

        # 2. 微专家路由
        router_little_logits = self.router_little(x)
        chosen_experts = torch.argmax(router_little_logits, dim=-1)

        # 送入全局遥测探针
        GLOBAL_TELEMETRY.record_step(weights_big, chosen_experts)

        # 选出主导专家做低秩残差注入
        top1_expert_id = chosen_experts[0, -1].item()
        lora_out = self.lora_pool[top1_expert_id](x)

        return big_out + 0.025 * lora_out


# ----------------------------------------------------------------------
# 3. 辅助函数：定位 Gemma 4 解码层
# ----------------------------------------------------------------------
def locate_gemma_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            return lm.layers
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "layers"):
            return lm.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    for name, module in model.named_modules():
        if name.endswith("language_model.layers") or name.endswith(
            "model.layers"
        ):
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                return module
    raise RuntimeError("无法定位 Gemma 4 解码器层结构！")


# ----------------------------------------------------------------------
# 4. 生成与全息透视战报输出
# ----------------------------------------------------------------------
def generate_and_diagnose(
    model,
    tokenizer,
    prompt,
    system_prompt="You are a helpful and versatile assistant.",
    max_new_tokens=350,
):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    print(f"\n👤 You: {prompt}\n")

    stop_ids = [tokenizer.eos_token_id]
    end_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if end_turn_id is not None and isinstance(end_turn_id, int):
        stop_ids.append(end_turn_id)

    # 重置遥测计数器并启动计时
    GLOBAL_TELEMETRY.reset()
    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.18,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    num_tokens = len(generated_tokens)
    speed = (num_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"🤖 Assistant: {response.strip()}\n")
    print(
        f"⚡ Speed: {speed:.1f} tokens/s ({num_tokens} tokens, {elapsed_ms:.0f} ms)"
    )

    # 计算双大核能耗百分比
    tot = (
        GLOBAL_TELEMETRY.arts_weights_sum + GLOBAL_TELEMETRY.sci_weights_sum
    ) or 1.0
    arts_pct = (GLOBAL_TELEMETRY.arts_weights_sum / tot) * 100.0
    sci_pct = (GLOBAL_TELEMETRY.sci_weights_sum / tot) * 100.0

    def make_bar(pct, length=20):
        filled = int(round((pct / 100.0) * length))
        return "█" * filled + " " * (length - filled)

    arts_bar = make_bar(arts_pct)
    sci_bar = make_bar(sci_pct)

    # 打印英文全息透视战报
    print("═" * 70)
    print("🌌【DualBigLittle-MoE: 1,120 Micro-Expert Holographic Telemetry】:")
    print(
        f"   🏛️  Tier-1 Arts Anchor Core:  {arts_pct:5.1f}% [{arts_bar}]"
    )
    print(
        f"   🔬 Tier-2 STEM Twin Core:    {sci_pct:5.1f}% [{sci_bar}]"
    )
    print("─" * 70)
    print("🪐【Top Active Specialist Clusters (35 Layers Aggregated)】:")

    top_active = GLOBAL_TELEMETRY.expert_activation_counter.most_common(5)
    if top_active:
        for exp_id, count in top_active:
            name = EXPERT_NAMES.get(exp_id, f"Expert_{exp_id}")
            print(f"   ✨ #{exp_id:02d} [{name:<18}]: {count:>6,d} calls")
    else:
        print("   (No micro-expert calls logged)")
    print("═" * 70)


# ----------------------------------------------------------------------
# 5. 主测试执行
# ----------------------------------------------------------------------
def main():
    model_id = "aifeifei798/Heretic-Scalpel-E2B"
    weights_path = "scalpel_dualbig_weights.pt"

    if not os.path.exists(weights_path):
        print(f"❌ Error: Weights file not found: {weights_path}")
        return

    print("=" * 70)
    print("🚀 Launching Heretic-Scalpel-E2B (1,120 Experts) Telemetry Shell...")
    print("=" * 70)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[*] Loading base backbone model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    # 剔除多模态组件防设备冲突
    for attr in [
        "vision_tower",
        "audio_tower",
        "visual",
        "audio",
        "mm_projector",
    ]:
        for parent in [model, getattr(model, "model", None)]:
            if parent is not None and hasattr(parent, attr):
                setattr(parent, attr, None)
    torch.cuda.empty_cache()

    layers = locate_gemma_layers(model)

    print("[*] Mounting Dual-Cores & 1,120 Micro-Experts (35 layers x 32)...")
    for layer in layers:
        layer.mlp = ScalpelDualBigInferenceWrapper(
            layer.mlp,
            hidden_dim=1536,
            rank=16,
            num_experts=32,
            device="cuda:0",
            dtype=dtype,
        )

    print(f"[*] Injecting trained organ weights from {weights_path} ...")
    weights = torch.load(weights_path, map_location="cuda:0")
    for i, layer in enumerate(layers):
        layer.mlp.big_sci.load_state_dict(weights[f"layer_{i}_big_sci"])
        layer.mlp.router_big.load_state_dict(weights[f"layer_{i}_router_big"])
        layer.mlp.router_little.load_state_dict(
            weights[f"layer_{i}_router_little"]
        )
        layer.mlp.lora_pool.load_state_dict(weights[f"layer_{i}_loras"])

    model.eval()
    print("[✔] 1,120 Micro-Expert Empire is fully loaded and ready!\n")

    # 预设的自动化双域基准用例 (对齐英文语境)
    test_cases = [
        "Mention these three things in a short suspenseful story: a rusty key, a flickering streetlamp, and a faint smell of lavender.",
        "Implement an optimized quicksort algorithm in Python, and explain its average and worst-case time complexity.",
        "Why is the sky blue on a clear sunny day? Explain using clear physical principles.",
        "Write a modern lyric poem in the first person expressing the loneliness and disorientation of standing at a crossroads on a rainy night.",
    ]

    print("🚀 [Phase 1: Automated Dual-Domain Benchmark]\n")
    for prompt in test_cases:
        generate_and_diagnose(model, tokenizer, prompt)

    print("\n🚀 [Phase 2: Interactive Terminal Chat] (Type 'exit' or 'q' to quit)")
    while True:
        try:
            user_input = input("\nEnter your prompt > ")
            if user_input.strip().lower() in ["exit", "q"]:
                break
            if not user_input.strip():
                continue
            generate_and_diagnose(model, tokenizer, user_input)
        except (KeyboardInterrupt, EOFError):
            break
    print("\nSession ended!")


if __name__ == "__main__":
    main()