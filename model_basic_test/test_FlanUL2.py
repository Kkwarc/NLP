import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
input_text = "Hello, what's your name?"

input_ids = tokenizer.encode(input_text, return_tensors="pt")
output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, num_beams=4)[0]

output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print(output_text)
