import json
import os
import shutil
import torch
from transformers import AutoTokenizer

EXPORT_DIR = "./Heretic-Scalpel-DualBig-2B"
BASE_MODEL_ID = "aifeifei798/Heretic-Scalpel-E2B"
WEIGHTS_PATH = "scalpel_dualbig_weights.pt"

os.makedirs(EXPORT_DIR, exist_ok=True)
print("=" * 70)
print(f"📦 启动【Heretic-Scalpel-DualBig-2B】Hugging Face 原生发布包制作...")
print("=" * 70)

# 清理 Hugging Face 本地动态模块缓存（防止读取修改前的旧代码缓存）
cache_dir = os.path.expanduser(
    "~/.cache/huggingface/modules/transformers_modules")
if os.path.exists(cache_dir):
    for item in os.listdir(cache_dir):
        if "Heretic" in item or "Scalpel" in item:
            shutil.rmtree(os.path.join(cache_dir, item), ignore_errors=True)

# ----------------------------------------------------------------------
# 1. 生成 configuration_scalpel.py
# ----------------------------------------------------------------------
config_py = """from transformers import PretrainedConfig

class ScalpelDualBigConfig(PretrainedConfig):
    model_type = "scalpel_dualbig"

    def __init__(
        self,
        base_model_name_or_path="aifeifei798/Heretic-Scalpel-E2B",
        hidden_dim=1536,
        rank=16,
        num_experts=32,
        gamma=0.025,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_experts = num_experts
        self.gamma = gamma
"""

with open(os.path.join(EXPORT_DIR, "configuration_scalpel.py"),
          "w",
          encoding="utf-8") as f:
    f.write(config_py)
print("[✔] 1. configuration_scalpel.py 写入成功！")

# ----------------------------------------------------------------------
# 2. 生成 modeling_scalpel.py (修复 base_model 属性名冲突)
# ----------------------------------------------------------------------
modeling_py = """import copy
import os
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModelForCausalLM
from .configuration_scalpel import ScalpelDualBigConfig

class GemmaLoRAMicroExpert(nn.Module):
    def __init__(self, hidden_dim=1536, rank=16, lora_alpha=16.0, device="cuda:0", dtype=torch.bfloat16):
        super().__init__()
        self.scaling = lora_alpha / rank
        self.lora_A = nn.Linear(hidden_dim, rank, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(rank, hidden_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling

class ScalpelDualBigWrapper(nn.Module):
    def __init__(self, original_mlp, hidden_dim=1536, rank=16, num_experts=32, gamma=0.025, device="cuda:0", dtype=torch.bfloat16):
        super().__init__()
        self.gamma = gamma
        self.big_arts = original_mlp
        self.big_sci = copy.deepcopy(original_mlp).to(device)
        self.router_big = nn.Linear(hidden_dim, 2, bias=False, device=device, dtype=dtype)
        self.router_little = nn.Linear(hidden_dim, num_experts, bias=False, device=device, dtype=dtype)
        self.lora_pool = nn.ModuleList([
            GemmaLoRAMicroExpert(hidden_dim=hidden_dim, rank=rank, device=device, dtype=dtype)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        router_big_logits = self.router_big(x)
        weights_big = torch.softmax(router_big_logits, dim=-1)
        arts_out = self.big_arts(x)
        sci_out = self.big_sci(x)
        big_out = (weights_big[..., 0:1] * arts_out) + (weights_big[..., 1:2] * sci_out)

        router_little_logits = self.router_little(x)
        chosen_expert = torch.argmax(router_little_logits.mean(dim=1), dim=-1).squeeze().item()
        lora_out = self.lora_pool[chosen_expert](x)

        return big_out + self.gamma * lora_out

class ScalpelDualBigForCausalLM(PreTrainedModel):
    config_class = ScalpelDualBigConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = None  # 修复：使用 self.model 替代保留只读属性 self.base_model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = ScalpelDualBigConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", torch.bfloat16))
        device_map = kwargs.pop("device_map", "cuda:0")

        model_instance = cls(config)
        print(f"[*] 正在挂载基础底座: {config.base_model_name_or_path} ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True
        )

        # 剔除未使用的多模态塔
        for attr in ["vision_tower", "audio_tower", "visual", "audio", "mm_projector"]:
            for parent in [base_model, getattr(base_model, "model", None)]:
                if parent is not None and hasattr(parent, attr):
                    setattr(parent, attr, None)
        torch.cuda.empty_cache()

        layers = None
        for path in [
            lambda m: m.model.language_model.layers,
            lambda m: m.language_model.layers,
            lambda m: m.model.layers,
        ]:
            try:
                layers = path(base_model)
                break
            except Exception:
                pass

        print(f"[*] 正在为 35 层装配双大核与 1,120 微专家 (gamma={config.gamma})...")
        device = next(base_model.parameters()).device
        for layer in layers:
            layer.mlp = ScalpelDualBigWrapper(
                layer.mlp,
                hidden_dim=config.hidden_dim,
                rank=config.rank,
                num_experts=config.num_experts,
                gamma=config.gamma,
                device=device,
                dtype=dtype
            )

        if os.path.isdir(pretrained_model_name_or_path):
            weights_file = os.path.join(pretrained_model_name_or_path, "scalpel_dualbig_weights.pt")
        else:
            from huggingface_hub import hf_hub_download
            weights_file = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="scalpel_dualbig_weights.pt")

        print(f"[*] 正在注入特训器官权重: {weights_file} ...")
        trained_weights = torch.load(weights_file, map_location=device)
        for i, layer in enumerate(layers):
            layer.mlp.big_sci.load_state_dict(trained_weights[f"layer_{i}_big_sci"])
            layer.mlp.router_big.load_state_dict(trained_weights[f"layer_{i}_router_big"])
            layer.mlp.router_little.load_state_dict(trained_weights[f"layer_{i}_router_little"])
            layer.mlp.lora_pool.load_state_dict(trained_weights[f"layer_{i}_loras"])

        base_model.eval()
        model_instance.model = base_model
        return model_instance

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    @property
    def device(self):
        return self.model.device
"""

with open(os.path.join(EXPORT_DIR, "modeling_scalpel.py"),
          "w",
          encoding="utf-8") as f:
    f.write(modeling_py)
print("[✔] 2. modeling_scalpel.py 写入成功！")

# ----------------------------------------------------------------------
# 3. 复制器官权重并保存 Tokenizer
# ----------------------------------------------------------------------
print("[*] 正在复制特训器官权重 scalpel_dualbig_weights.pt (约 1.6 GB)...")
shutil.copy(WEIGHTS_PATH, os.path.join(EXPORT_DIR,
                                       "scalpel_dualbig_weights.pt"))
print("[✔] 3. scalpel_dualbig_weights.pt 复制成功！")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID,
                                          trust_remote_code=True)
tokenizer.save_pretrained(EXPORT_DIR)
print("[✔] 4. Tokenizer 配置文件保存成功！")

# ----------------------------------------------------------------------
# 4. 生成带核心 auto_map 的 config.json
# ----------------------------------------------------------------------
config_dict = {
    "architectures": ["ScalpelDualBigForCausalLM"],
    "model_type": "scalpel_dualbig",
    "base_model_name_or_path": BASE_MODEL_ID,
    "hidden_dim": 1536,
    "rank": 16,
    "num_experts": 32,
    "gamma": 0.025,
    "auto_map": {
        "AutoConfig": "configuration_scalpel.ScalpelDualBigConfig",
        "AutoModelForCausalLM": "modeling_scalpel.ScalpelDualBigForCausalLM",
    },
}

with open(os.path.join(EXPORT_DIR, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config_dict, f, indent=2, ensure_ascii=False)
print("[✔] 5. config.json 写入成功！")

# ----------------------------------------------------------------------
# 5. 生成标准 README.md
# ----------------------------------------------------------------------
readme_content = """---
language:
- en
- zh
tags:
- moe
- hierarchical-moe
- dual-core
- gemma4
license: apache-2.0
---

# Heretic-Scalpel-DualBig-2B

A Hierarchical Dual-Core MoE (Gemma 4 text backbone) with 1,120 micro-experts and $\\gamma = 0.025$ scaling.

## Quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "your-username/Heretic-Scalpel-DualBig-2B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True
)

prompt = "以‘深山古刹残钟’为题，写一段充满禅意与寂寥的白话散文。"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```
"""
with open(os.path.join(EXPORT_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)
print("[✔] 6. README.md 生成成功！")
print("=" * 70)
print(f"🎉 全部封装完毕！目录位于: {EXPORT_DIR}")
print("=" * 70)
