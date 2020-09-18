import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

class CNN(nn.Module):
    def __init__(self):
        # Input Image Shape: (28, 28, 1)
        super(CNN, self).__init__()
        # input channel size: 1 / num of conv1 output channel: 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.lin1 = nn.Linear(3*3*64, 64)
        self.lin2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3*3*64)
        x = F.relu(self.lin1(x))
        x = F.softmax(self.lin2(x), dim=1)
        return x


"""
from torchsummary import summary
model = CNN()
summary(model, (1, 28, 28))
"""


def train_test_split(X, y, ratio, shuffle=True):
    length = X.shape[0]
    if shuffle:
        idx = np.random.permutation(length)
    else:
        idx = np.arange(length)
    split = int((ratio) * length)
    X_train, y_train = X[idx[:split], ...], y[idx[:split]]
    X_test, y_test = X[idx[split:], ...], y[idx[split:]]
    return X_train, y_train, X_test, y_test


mnist = fetch_openml("mnist_784")
X_784, y_784 = mnist.data, mnist.target
X_784 /= 255  # scaling
X_784 = X_784.reshape(-1, 1, 28, 28)
y_784 = np.array(list(map(int, y_784)))
split_ratio = 0.7
X_train, y_train, X_test, y_test = train_test_split(X_784, y_784, split_ratio)

learning_rate = 0.001
num_epoch = 5
batch_size = 64
X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train).long()

train_dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = CNN()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()
loss_arr = []

for i in range(num_epoch):
    loss_log = []
    for batch_idx, samples in enumerate(dataloader):
        X_batch, y_batch = samples
        optimizer.zero_grad()

        y_pred = model(X_batch.float())
        loss = loss_func(y_pred, y_batch)
        loss_log.append(loss.item())
        loss.backward()
        optimizer.step()
        print("Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}".format(
            i + 1, num_epoch, batch_idx + 1, len(dataloader), loss.item()
        ))
    loss_arr.append(loss_log)


with torch.no_grad():
    X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test)
    y_pred = model(X_test)
    correct_prediction = torch.argmax(y_pred, 1) == y_test
    accuracy = correct_prediction.float().mean()
    print("Accuracy: {0:.3f}".format(accuracy.item()))

