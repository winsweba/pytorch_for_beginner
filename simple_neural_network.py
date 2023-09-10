import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transform





#Creating Fully neural network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#Test case
# model= NN(784, 10)
# x = torch.randn(64, 784)
# print(model(x).shape)

# Set device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameter
input_size = 784
num_classes = 10
learning_rete = 0.001
batch_size = 64
nun_epochs = 2

#Loading Data
train_dataset = datasets.MNIST(root='datasets/', train=True, transform=transform.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 

test_dataset = datasets.MNIST(root='datasets/', train=False, transform=transform.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#initializing networks
model = NN(input_size=input_size, num_classes=num_classes).to(device=device)

#loss and OPtimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),learning_rete )

for epoch in range(nun_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):

        #get dat to cuda if possible
        data = data.to(device)
        targets = targets.to(device)
        # get to correct shape
        data = data.reshape(data.shape[0], -1)
        
        #forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or adam step
        optimizer.step() 


#check accuracy on train and test data
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("checking accuracy on training data")
    else:
        print("checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, prediction =  scores.max(1)
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)

        print(f'get{num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)* 100:.2f}')

    model.train()
    

check_accuracy(train_loader, model=model)
check_accuracy(test_loader, model=model)
