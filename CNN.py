import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

# Convert images into tensors and normalize them
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])

# Load dataset into train and test groups
train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transform_train, download = True)
test_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transform_test, download = True)

# Create minibatch
train_batch = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)
test_batch = DataLoader(dataset = test_dataset, batch_size = 64)

# CNN
class CNN(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(1600, 256)
        self.linear2 = nn.Linear(256, 10)
    
    def forward(self, x) :
        x = self.pool(self.relu(self.conv1(x))) 
        x = self.pool(self.relu(self.conv2(x))) 
        x = torch.flatten(x,1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
model = CNN()

# Cross-entropy loss function
loss_function = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(params = model.parameters(), lr = 0.001)

# Training 
num_epochs = 5

for epoch in range(num_epochs) :
    model.train()
    running_loss = 0.0

    for images, labels in train_batch :

        # Reset gradients
        optimizer.zero_grad()

        # Predictions
        outputs = model(images)

        # Loss calcul
        batch_loss = loss_function(outputs, labels)

        # Back prop
        batch_loss.backward()

        # Udpate of weights
        optimizer.step()
        
        running_loss += batch_loss.item()

    avg_loss = running_loss/len(train_batch)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "cnn_model.pth")

# Testing
model.eval()
count_answers = 0
total = 0

# Get rid of gradients
with torch.no_grad() :
    for images, labels in test_batch :
        outputs = model(images)

        # Get highest score
        prediction = torch.max(outputs,1)[1]

        # Count the amount of good answers by comparing prediction to the label 
        count_answers += (prediction == labels).sum().item()

        # Count th enumber of images by batch
        total += labels.size(0)
    accuracy = count_answers/total
print(f"Accuracy : {accuracy*100:.2f}%")

