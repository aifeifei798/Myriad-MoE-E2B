
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "./Heretic-Scalpel-DualBig-2B"

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