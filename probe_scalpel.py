import torch
from transformers import AutoConfig, AutoModelForCausalLM

model_id = "aifeifei798/Heretic-Scalpel-E2B"
print(f"[*] 正在探测你的亲儿子模型: {model_id} ...")

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
print(f"    - 模型架构 (Architectures): {config.architectures}")
print(f"    - 模型类型 (Model Type)   : {config.model_type}")

# 探测文本骨干参数
text_cfg = getattr(config, "text_config", config)
hidden_size = getattr(text_cfg, "hidden_size", 1536)
num_layers = getattr(text_cfg, "num_hidden_layers", 35)

print(f"    - 隐藏层维度 (Hidden Size) : {hidden_size}")
print(f"    - 解码器层数 (Num Layers) : {num_layers}")
print(
    f"    - 单微专家体积 (Rank-16)  : {(hidden_size * 16 * 2 * 2) / 1024:.1f} KB"
)
print("\n[✔] 探测成功！准备开始做双大核器官移植！")
