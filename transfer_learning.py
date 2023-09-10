import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transform
import torchvision.models as models



# Hyperparameter
in_channels = 3
num_classes = 10
learning_rete = 1e-3
batch_size = 1024
nun_epochs = 5
# load_model = True

import sys

# Set device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 10),)
# model.classifier = nn.Linear(512, 10)
model.to(device=device)

# for i in range(1,7):
#     model.classifier[i] = Identity()


# print(model)



# sys.exit()





#Loading Data
train_dataset = datasets.CIFAR10(root='datasets/cf10/', train=True, transform=transform.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 

# test_dataset = datasets.MNIST(root='datasets/', train=False, transform=transform.ToTensor(), download=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)



#loss and OPtimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),learning_rete )




for epoch in range(nun_epochs):
    losses = []
   
    for batch_idx, (data, targets) in enumerate(train_loader):

        #get dat to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
   
        
        #forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or adam step
        optimizer.step() 

    mean_loss = sum(losses)/len(losses)
    print(f'Loss at epoch{epoch} was {mean_loss:.5f}')





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

            scores = model(x)
            _, prediction =  scores.max(1)
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)

        print(f'get{num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)* 100:.2f}')

    model.train()
    

# check_accuracy(train_loader, model=model)
# check_accuracy(test_loader, model=model)
