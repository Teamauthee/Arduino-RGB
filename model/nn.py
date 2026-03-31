import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ColorClassifier, self).__init__()

        hidden_1 = 16
        hidden_2 = 8

        self.fc1 = nn.Linear(input_size, hidden_1)
        self.bn1 = nn.BatchNorm1d(hidden_1)

        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.bn2 = nn.BatchNorm1d(hidden_2)

        self.fc3 = nn.Linear(hidden_2, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = self.fc3(x)
        return x