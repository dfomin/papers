from torch import nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm = nn.LocalResponseNorm(5, 1e-4, 0.75, 2)
        self.pool = nn.MaxPool2d(3, 2)
        self.dropout = nn.Dropout(p = 0.5)

        self.conv1 = nn.Conv2d(3, 96, 11, 4, 3)
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)

        self.linear1 = nn.Linear(256 * 6 * 6, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.pool(self.norm(F.relu(self.conv1(x))))
        x = self.pool(self.norm(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1)
        x = self.dropout(self.linear1(x))
        x = self.dropout(self.linear2(x))
        x = F.softmax(self.linear3(x), dim=0)
        return x
