import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN6(nn.Module):
    def __init__(self, num_classes=7):
        """
        CNN6 model for audio classification.
        Args:
            num_classes (int): Number of output classes.
        """
        super(CNN6, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # Input: 1 channel
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        
        # Define pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layer
        self.fc = None  # Placeholder, will initialize dynamically in forward
        self.num_classes = num_classes

    def forward(self, x):
        # Apply convolutions and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))

        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        if self.fc is None:
            self.fc = nn.Linear(x.size(1), self.num_classes).to(x.device)

        x = self.fc(x)
        return x
        

class CNN6_6(nn.Module):
    def __init__(self, num_classes):
        super(CNN6_6, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)