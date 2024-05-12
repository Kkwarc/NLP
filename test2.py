import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

def transform_labels(row):
    x = {"aristotle": 0,
         "schopenhauer": 1,
         "nietzsche": 2,
         "hegel": 3,
         "kant": 4,
         "sartre": 5,
         "plato": 6,
         "freud": 7,
         "spinoza": 8}
    return x[row]


df = pd.read_csv("data_set.csv", sep="@")
df["label"] = df["author"].apply(transform_labels)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
bert_model = BertModel.from_pretrained('bert-base-uncased')


def tokenize_data(data):
    # token = bert_tokenizer.batch_encode_plus(data, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True, max_length=128, return_tensors='pt')
    encoded_input = bert_tokenizer(data.to_list(), return_tensors='pt', add_special_tokens=False, pad_to_max_length=True, max_length=10)
    output = bert_model(**encoded_input)
    text_embedding = output.last_hidden_state[0]
    middle_tensor = text_embedding[1]
    return middle_tensor


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, inputs, attention_mask):
        x = inputs.float()  # Cast input to float
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, criterion, optimizer, scheduler, train_loader, test_loader, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, attention_mask, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            correct = 0
            total = 0
            for inputs, attention_mask, labels in test_loader:
                outputs = model(inputs, attention_mask)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss += criterion(outputs, labels).item()

        print(f"Test Loss: {test_loss}, Accuracy: {100 * correct / total}")
        scheduler.step(test_loss)

X_train, X_test, y_train, y_test = train_test_split(df["quote"], df["label"], test_size=0.2, random_state=42)

X_train_tokens_attention = tokenize_data(X_train)
X_train_tokens = X_train_tokens_attention['input_ids']
X_train_attention = X_train_tokens_attention['attention_mask']

X_test_tokens_attention = tokenize_data(X_test)
X_test_tokens = X_test_tokens_attention['input_ids']
X_test_attention = X_test_tokens_attention['attention_mask']


y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tokens, X_train_attention, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test_tokens, X_test_attention, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_dim = 128
hidden_dim1 = 512
hidden_dim2 = 256
output_dim = 9
model = MLP(input_dim, hidden_dim1, hidden_dim2, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

train_model(model, criterion, optimizer, scheduler, train_loader, test_loader, num_epochs=50)

model.eval()
with torch.no_grad():
    outputs = []
    for inputs, attention_mask, _ in test_loader:
        outputs.extend(model(inputs, attention_mask))
    outputs = torch.stack(outputs)
    _, predicted = torch.max(outputs, 1)
    accuracy = accuracy_score(y_test, predicted.numpy())
    cm = confusion_matrix(y_test, predicted.numpy())

class_names = ["aristotle", "schopenhauer", "nietzsche", "hegel", "kant", "sartre", "plato", "freud", "spinoza"]

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()