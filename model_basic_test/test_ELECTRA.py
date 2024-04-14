import torch
from transformers import ElectraForSequenceClassification, ElectraTokenizer
model = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator")

tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
input_text = "Cześć, jak masz na imię?"

input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model(input_ids)[0]
predicted_class = torch.argmax(output, dim=1).item()

print(f"Tekst został zaklasyfikowany jako klasa {predicted_class}.")
