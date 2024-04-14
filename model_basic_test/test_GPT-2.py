import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

input_text = "Cześć, jak masz na imię?"

input_ids = tokenizer.encode(input_text, return_tensors="pt")
output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, num_beams=4)[0]

output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print(output_text)
