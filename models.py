import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Clasificator(nn.Module):
    def __init__(self):
        super().__init__()
        #125
        self.conv1 = nn.Conv1d(100, 90, 4, dtype=torch.double)
        #122
        self.pool1 = nn.MaxPool1d(2)
        #61
        self.conv2 = nn.Conv1d(90, 70, 2, dtype=torch.double)
        #60
        self.pool2 = nn.MaxPool1d(2)
        self.linear1 = nn.Linear(30*70, 100, dtype=torch.double)
        self.linear2 = nn.Linear(100, 9, dtype=torch.double)
        self.dropout1 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.dropout1(self.linear1(x)))
        x = self.linear2(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1, dtype=torch.double)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2, dtype=torch.double)
        self.fc3 = nn.Linear(hidden_dim2, output_dim, dtype=torch.double)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LSTM__Clasificator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = torch.device('cpu')
        self.lstm = nn.LSTM(input_size = 100, hidden_size = hidden_size,
                             batch_first=True, dtype = torch.double)
        self.linear1 = nn.Linear(hidden_size, hidden_size, dtype=torch.double)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(hidden_size, 9, dtype=torch.double)

    def prep(self, batch_size):
        self.hidden = torch.zeros(1, batch_size, self.hidden_size,
                                   dtype=torch.double).to(self.device)
        self.state = torch.zeros(1, batch_size, self.hidden_size
                                 , dtype=torch.double).to(self.device)

    def forward(self, x):
        all_outputs, (self.hidden, self.state) = self.lstm(x, (self.hidden, self.state))
        all_outputs = self.linear2(F.relu(self.dropout(self.linear1(all_outputs))))
        return all_outputs

    def to(self, device):
        a = super().to(device)
        self.device = device
        return a
