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
        
    

class WeakCNN(nn.Module):
    def __init__(self, num_classes=2):
        """
        Weak CNN model for audio classification.
        Args:
            num_classes (int): Number of output classes.
        """
        super(WeakCNN, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # Input: 1 channel
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Define pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layer
        self.fc = None
        self.num_classes = num_classes

    def forward(self, x):
        # Apply convolutions and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layer
        if self.fc is None:
            self.fc = nn.Linear(x.size(1), self.num_classes).to(x.device)
        x = self.fc(x)
        return x
    
    def get_weights(self):
        return [val.cpu().numpy() for _, val in self.state_dict().items()]