import torch
from transformers import BertForSequenceClassification, BertTokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
input_text = "Cześć, jak masz na imię?"


input_ids = tokenizer.encode(input_text, return_tensors="pt")
print(input_ids)
output = model(input_ids)[0]
print(output)
predicted_class = torch.argmax(output, dim=1).item()

print(f"Tekst został zaklasyfikowany jako klasa {predicted_class}.")
