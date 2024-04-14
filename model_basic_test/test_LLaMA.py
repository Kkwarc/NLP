import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("path/to/llama/model")
tokenizer = LlamaTokenizer.from_pretrained("path/to/llama/tokenizer")

input_text = "Cześć, jak masz na imię?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, num_beams=4)[0]

output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print(output_text)
