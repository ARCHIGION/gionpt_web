from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map="auto",
                                             torch_dtype="auto",
                                             trust_remote_code=True)

prompt = "한국의 전통 건축에서 처마의 기능은 무엇인가요?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
