import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


class TwoLayersNet(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(TwoLayersNet, self).__init__()
        self.lin1 = nn.Linear(input_shape, 512)
        self.lin2 = nn.Linear(512, output_shape)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.softmax(self.lin2(x))
        return x


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
y_784 = np.array(list(map(int, y_784)))
split_ratio = 0.7
X_train, y_train, X_test, y_test = train_test_split(X_784, y_784, split_ratio)

learning_rate = 0.001
num_epoch = 5
batch_size = 128
X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train).long()

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

model = TwoLayersNet(X_train.shape[1], len(np.unique(y_train)))
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()
loss_arr = []
# 모델 훈련
for _ in range(num_epoch):
    for batch_idx, samples in enumerate(dataloader):
        X_batch, y_batch = samples
        y_pred = model(X_batch.float())
        loss = loss_func(y_pred, y_batch)
        loss_arr.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}".format(
            _+1, num_epoch, batch_idx+1, len(dataloader), loss.item()
        ))


with torch.no_grad():
    # 훈련된 모델 테스트
    X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test)
    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == y_test
    accuracy = correct_prediction.float().mean()
    print("Accuracy: {0:.3f}". format(accuracy.item()))
