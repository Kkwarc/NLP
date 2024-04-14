import torch
from transformers import BartForConditionalGeneration, BartTokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
input_text = "Hello, what's your name?"

input_ids = tokenizer.encode(input_text, return_tensors="pt")
print(input_ids)
output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, num_beams=4)[0]
print(output_ids)
output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print(output_text)
