import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1   = nn.Linear(320, 50)
        self.fc2   = nn.Linear(50, 10)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.relu      = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = self.relu(self.maxpool2d(self.conv1(x)))
        x = self.relu(self.maxpool2d(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.log_softmax(x)