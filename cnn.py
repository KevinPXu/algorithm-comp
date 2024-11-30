import torch
import torch.nn as nn

class cnn(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)


        self.fc1 = nn.Linear(7 * 7 * 128, 1024)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.2)

        self.out = nn.Linear(512, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))   # Output: [batch_size, 32, 20, 20]
        x = self.relu(self.bn2(self.conv2(x)))   # Output: [batch_size, 64, 9, 9]
        x = self.relu(self.bn3(self.conv3(x)))  
         # Output: [batch_size, 64, 7, 7]
        x = x.view(x.size(0), -1)      # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.out(x)
