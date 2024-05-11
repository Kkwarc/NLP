import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Clasificator(nn.Module):
    def __init__(self):
        super().__init__()
        #125
        self.conv1 = nn.Conv1d(100, 150, 4, dtype=torch.double)
        #122
        self.pool1 = nn.MaxPool1d(2)
        #61
        self.conv2 = nn.Conv1d(150, 50, 2, dtype=torch.double)
        #60
        self.pool2 = nn.MaxPool1d(2)
        self.linear1 = nn.Linear(30*50, 200, dtype=torch.double)
        self.linear2 = nn.Linear(200, 9, dtype=torch.double)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class LSTM__Clasificator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = torch.device('cpu')
        self.lstm = nn.LSTM(input_size = 100, hidden_size = hidden_size,
                             batch_first=True, dtype = torch.double)
        self.linear = nn.Linear(hidden_size, 9, dtype=torch.double)

    def prep(self, batch_size):
        self.hidden = torch.zeros(1, batch_size, self.hidden_size,
                                   dtype=torch.double).to(self.device)
        self.state = torch.zeros(1, batch_size, self.hidden_size
                                 , dtype=torch.double).to(self.device)

    def forward(self, x):
        all_outputs, (self.hidden, self.state) = self.lstm(x, (self.hidden, self.state))
        all_outputs = self.linear(all_outputs)
        return all_outputs
    
    def to(self, device):
        a = super().to(device)
        self.device = device
        return a