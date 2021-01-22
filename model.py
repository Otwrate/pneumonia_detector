import torch.nn as nn
import torch.nn.functional as F
import torch

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.cov_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cov_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cov_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.d_1 = nn.Dropout(0.25)
        self.d_2 = nn.Dropout(0.5)
        self.d_3 = nn.Dropout(0.5)
        self.fc_1 = nn.Linear(86528, 1028)
        self.fc_2 = nn.Linear(1028, 512)
        self.fc_3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.cov_1(x)
        x = F.relu(x)
        x = self.pool_1(x)

        x = self.cov_2(x)
        x = F.relu(x)
        x = self.pool_2(x)

        x = self.cov_3(x)
        x = F.relu(x)
        x = self.pool_3(x)

        x = torch.flatten(x, start_dim=1)


        x = self.fc_1(x)
        x = F.relu(x)
        x = self.d_1(x)


        x = self.fc_2(x)
        x = F.relu(x)
        x = self.d_2(x)


        x = self.fc_3(x)
        x = F.relu(x)

        output = torch.log_softmax(x, dim=1)

        return output
