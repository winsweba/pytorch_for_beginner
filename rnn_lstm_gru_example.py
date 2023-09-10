import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transform

#TODO Read this Line
""" 
#TODO NB. RNN and GRU code look similar
## Make Sure to Read ALL the  COMMENT in this code Some of the Lines are Ether LSTM, GRU, RNN 

"""





#Test case
# model= NN(784, 10)
# x = torch.randn(64, 784)
# print(model(x).shape)

# Set device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameter
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256 
num_classes = 10
learning_rete = 0.001
batch_size = 64
nun_epochs = 2


# Create a  RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #TO RNN
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        
        # TO GRU
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first = True)
        
        # TO LSTM
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        
        ## Comment this code for LSTM To Work
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)
        
        # Only Need for LSTM
        # self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)
        ## only Need cIn LSTM
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)

        # Forward Prop

        #TO RNN
        out, _ = self.rnn(x, h0)

        # TO GRU
        # out, _ = self.gru(x, h0)
        
        # TO LSTM
        # out, _ = self.lstm(x, (h0,c0))

        ## Comment this code for LSTM To Work
        out = out.reshape(out.shape[0], -1)

        ## Comment this code for LSTM To Work
        out = self.fc(out)


        ## only Need cIn LSTM
        # out = self.fc(out[:, -1,:])

        return out 





#Loading Data
train_dataset = datasets.MNIST(root='datasets/', train=True, transform=transform.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 

test_dataset = datasets.MNIST(root='datasets/', train=False, transform=transform.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#initializing networks
model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes).to(device=device)

#loss and OPtimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),learning_rete )

for epoch in range(nun_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):

        #get dat to cuda if possible
        data = data.to(device).squeeze(1)
        targets = targets.to(device)
       
        
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
            x = x.to(device).squeeze(1)
            y = y.to(device)

            scores = model(x)
            _, prediction =  scores.max(1)
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)

        print(f'get{num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)* 100:.2f}')

    model.train()
    

check_accuracy(train_loader, model=model)
check_accuracy(test_loader, model=model)
