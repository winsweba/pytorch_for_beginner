import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transform
import torchvision.models as models

from custom_data import CatsAndDogsDataset



# Hyperparameter
in_channels = 3
num_classes = 10
learning_rete = 1e-3
batch_size = 32
nun_epochs = 5
# load_model = True



# Set device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

path_to_csv = r"C:\Users\winsweb\personal_project\python\ml\pytorch\neural_networks\custom_dataset\cats_dogs.csv"
path_to_img = r"C:\Users\winsweb\personal_project\python\ml\pytorch\neural_networks\custom_dataset\cats_dogs_resized"
# Loading the Data
dataset = CatsAndDogsDataset(csv_file=path_to_csv, root_dir=path_to_img,
                             transform=transform.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [7, 3])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)





# Model
model = models.googlenet(pretrained = True)
model.to(device=device)

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
    

print("Checking Accuracy on Train set")
check_accuracy(train_loader, model=model)
print("Checking Accuracy on Test set")
check_accuracy(test_loader, model=model)
