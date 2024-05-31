"""
author: rohan singh
simple pytorch model to train a classifier for the FashionMNIST dataset
"""



# imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor



# downloading the training data from open datasets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# downloading the test data from open datasets
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)



# creating the dataloader
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)



# setting up the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)



# defining the model
class FashionNeuralNetwork(nn.Module):


    # initialization function (defining the neural network model)
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    
    # defining the forward pass
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    


# instantiating a model
model = FashionNeuralNetwork().to(device=device)



# defining optimization parameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)



# defining the training loop
def train(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)
    model.train()

    # iterating through the dataset
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # computing the loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropogation step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch%100 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"loss: {loss:>5f} [{current:>5d}/{size:5d}]")



# testing function to check the models performance
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"test error: \n accuracy: {(100*correct):>0.1f}%, avg loss: {test_loss:>5f} \n")



# training the model
print("\nTraining\n-------------------------------")
epochs = 5
for t in range(epochs):
    print(f"epoch {t+1}\n-------------------------------")
    train(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
    test(dataloader=test_dataloader, model=model, loss_fn=loss_fn)
print("\nTraining Complete\n-------------------------------\n")



# main function for demonstration
def main():

    # using the known labels for the FashionMNIST dataset
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot","Emile"]
    
    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'predicted: "{predicted}", actual: "{actual}"')



main()







