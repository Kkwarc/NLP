import torch
import torch.nn as nn
import torch.optim as optim
from models import MLP
from get_embedded_data import get_data_BERT_MLP
device = torch.device("cpu")
dataloader = get_data_BERT_MLP(200)
network = MLP(input_dim=128, hidden_dim1=512, hidden_dim2=256, output_dim=9).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters())
max_epoch = 30
network.train()
for epoch in range(max_epoch):

    running_loss = 0.0
    correct = 0
    total = 0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    print('[%d/%d] loss: %.3f accuracy: %d' %
          (epoch + 1, max_epoch, running_loss / 2000, 100 * correct / total))
    running_loss = 0.0

print('Finished Training')