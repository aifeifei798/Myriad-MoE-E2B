import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "./Heretic-Scalpel-DualBig-E2B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True
)

prompt = "用 C++20 实现一个无锁环形缓冲区（Lock-Free Ring Buffer），要求严格基于 std::atomic 内存序（acquire/release 语义），必须处理多生产者单消费者的缓存行伪共享（Cache False Sharing）问题，并给出内存对齐代码。"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

# 1. 统计前重置探针
model.reset_stats()

# 2. 计时生成
t0 = time.perf_counter()
outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.7)
elapsed_sec = time.perf_counter() - t0

# 3. 计算速度与字数
gen_tokens_ids = outputs[0][inputs.input_ids.shape[1]:]
gen_tokens = len(gen_tokens_ids)
speed = gen_tokens / elapsed_sec if elapsed_sec > 0 else 0

# 4. 打印回复内容
print("\n🤖 Assistant:\n")
print(tokenizer.decode(gen_tokens_ids, skip_special_tokens=True))

# 5. 打印全息仪表盘
model.show_dashboard(speed=speed, gen_tokens=gen_tokens, elapsed_sec=elapsed_sec)