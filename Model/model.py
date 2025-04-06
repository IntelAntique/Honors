import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
from json import loads
import pandas as pd
from torchmetrics import ROC
import matplotlib.pyplot as plt
from torchmetrics.functional import auroc
from model_lib import roc 

with open("output.txt", "r") as input:
    lines = input.read().splitlines()

data = []
for line in lines:
    x = loads(line)
    data.append(x[0]['Answers'])

dataset = pd.read_csv("dataset/Phishing_Email.csv", skiprows=1, header=None)

labels = []
for value in dataset[2].values: # Phishing 7328, Safe 11322
    labels.append(0 if value == "Safe Email" else 1)

# print(labels[:2000].count(1)) # Phishing 822, Safe 1178

for i in range(len(data)):
    data[i].append(labels[i])

training_num = 1800

batch_size = 200
training_data = torch.tensor(data[:training_num])
test_data = torch.tensor(data[training_num:]) # individual test data, not a list

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


n = 12 # questions
# for data in test_dataloader:
#     print(data)
#     X, y = data[:, : n], data[:, n]
#     print(f"Shape of X [N, C]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

input = 12
output = 2

class BinaryClassification(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output),
        )


    def forward(self, x):
        return self.sequential(x)

model = BinaryClassification(input, output).to(device)
loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, data in enumerate(dataloader):
        X, y = data[:, : n].type(torch.float).to(device), data[:, n].type(torch.long).to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % batch_size == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data in dataloader:
            X, y = data[:, : n].type(torch.float).to(device), data[:, n].type(torch.long).to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
roc(test_dataloader, model, device, n)