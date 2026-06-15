import os

import torch
import torch.nn as nn

# This allows to use the model without training it everytime

class CNN(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32,64,3)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(1600,256)
        self.linear2 = nn.Linear(256,10)
    def forward(self,x) :
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

model = CNN()
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "cnn_model.pth") # Get the weigth from the already trained model 
model.load_state_dict(torch.load(model_path))
model.eval()