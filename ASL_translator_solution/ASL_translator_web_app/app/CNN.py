from torch import nn
import torch
class myCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,15,3)
        self.fc1 = nn.Linear(15*34*34,100)
        self.fc2 = nn.Linear(100,60)
        self.fc3 = nn.Linear(60,28)
        self.relu = nn.ReLU()

    def forward(self,x):
        # first convolution
        x = self.conv1(x)
        x = self.relu(x)
        
        x=self.pool(x)

        # second convolution
        x = self.conv2(x)
        x = self.relu(x)
                
        # fully connected
        x = torch.flatten(x,1) # flatten all dimensions except the batch

        # fc1
        x = self.fc1(x)
        x = self.relu(x)

        # fc2
        x = self.fc2(x)
        x = self.relu(x)

        # fc out
        x = self.fc3(x)

        return x
