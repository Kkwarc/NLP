from transformers import BertTokenizer
import torch
import torch.nn as nn

# BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize(text):
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )


# MLP model
class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Tokenize and create dataset
texts = [...]  # list of text samples
input_ids = [tokenize(text)['input_ids'].squeeze() for text in texts]
attention_masks = [tokenize(text)['attention_mask'].squeeze() for text in texts]
labels = torch.tensor([...])  # list of labels

dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
input_dim = 768  # BERT output dim
hidden_dim = 256
output_dim = len(set(labels.tolist()))  # num classes
model = TextClassifier(input_dim, hidden_dim, output_dim)

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

for epoch in range(num_epochs):
    for input_ids, attention_masks, labels in dataloader:
        optimizer.zero_grad()


        logits = model(bert_output)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

# Evaluate model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for input_ids, attention_masks, labels in dataloader:
        bert_output = bert_model(input_ids, attention_mask=attention_masks)[1]
        logits = model(bert_output)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

print(f'Accuracy: {correct / total:.4f}')
