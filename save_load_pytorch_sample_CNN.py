import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transform



# Hyperparameter
in_channels = 1
num_classes = 10
learning_rete = 0.001
batch_size = 64
nun_epochs = 10
load_model = True



# Create a RNN
class CNN(nn.Module):
    def __init__(self,in_channels=1, num_classes =10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

#Test case CNN
# model= CNN()
# x = torch.randn(64,1,28, 28)
# print(model(x).shape)



model_path = 'model_savings/my_save_checkpoint.pth.tar'


# Save Checkpoint
def save_checkpoint(state, filename=model_path):
    print ("=> Saving Checkpoint")
    torch.save(state, filename)

# Model Savings
def load_checkpoint(checkpoint):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


# Set device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#Loading Data
train_dataset = datasets.MNIST(root='datasets/', train=True, transform=transform.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 

test_dataset = datasets.MNIST(root='datasets/', train=False, transform=transform.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#initializing networks
model = CNN().to(device=device)

#loss and OPtimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),learning_rete )

# now Loading the model
if load_model:
    load_checkpoint(torch.load(model_path))

for epoch in range(nun_epochs):
    losses = []
    if epoch % 3 == 0:
        checkpoint = {"state_dict": model.state_dict(), 'optimizer':  optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_loader):

        #get dat to cuda if possible
        data = data.to(device)
        targets = targets.to(device)
   
        
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
