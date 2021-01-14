import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

class MNIST_Logistic_Regression(nn.Module):
    def __init__(self):
        super().__init__()
        #self.lin = nn.Linear(784, 10) linear model
        self.lin = nn.Linear(784, 50)
        self.relu = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(50,10)

    def forward(self, x):
        #apply deep
        out = self.lin(x)
        out = self.relu(out)
        out = self.lin2(out)
        return out

# Load the data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

## Training
# Instantiate model
#handles all the W and b initialization
model = MNIST_Logistic_Regression()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss() #equivlanet to F.cross_entropy
#this is telling the model that we need to keep track of all models
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Iterate through train set minibatchs
#train the model 5 times
for epoch in range(5):
  for images, labels in tqdm(train_loader):
      # Zero out the gradients
      optimizer.zero_grad()
      
      # Forward pass
      x = images.view(-1, 28*28)
      y = model(x)
      loss = criterion(y, labels) #cross entropy loss
      # Backward pass
      loss.backward()
      optimizer.step()

## Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    # Iterate through test set minibatchs 
    for images, labels in tqdm(test_loader):
        # Forward pass
        x = images.view(-1, 28*28)
        y = model(x)
        
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())
    
print('Test accuracy: {}'.format(correct/total))
