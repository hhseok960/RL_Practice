import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import numpy as np
from sklearn.datasets import load_boston


class DNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(DNN, self).__init__()
        self.lin1 = nn.Linear(input_shape, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, output_shape)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


def train_test_split(X, y, ratio, shuffle=True):
    length = X.shape[0]
    if shuffle:
        idx = np.random.permutation(length)
    else:
        idx = np.arange(length)
    split = int(ratio * length)
    X_train, y_train = X[idx[:split], ...], y[idx[:split]]
    X_test, y_test = X[idx[split:], ...], y[idx[split:]]
    return X_train, y_train, X_test, y_test


boston = load_boston()
X, y = boston.data, boston.target
ratio = 0.8
X_train, y_train, X_test, y_test = train_test_split(X, y, ratio)
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train_scale = (X_train - mean) / std
X_test_scale = (X_test - mean) / std

k_fold = 4
learning_rate = 0.001
num_epoch = 50
batch_size = 32

kfold_loss = []

split_idx = [X_train.shape[0] * i / k_fold for i in range(k_fold + 1)]
split_idx = list(map(int, split_idx))
for i in range(k_fold):
    X_train_fold = np.delete(X_train_scale, np.arange(split_idx[0], split_idx[1]), axis=0)
    y_train_fold = np.delete(y_train, np.arange(split_idx[0], split_idx[1]), axis=0)
    X_valid = X_train_scale[split_idx[i]:split_idx[i + 1]]
    y_valid = y_train[split_idx[i]:split_idx[i + 1]]
    fold_dataset = TensorDataset(torch.from_numpy(X_train_fold).float(), torch.from_numpy(y_train_fold).float())
    fold_dataloader = DataLoader(fold_dataset, batch_size=batch_size, shuffle=True)

    model = DNN(X_train.shape[1], 1)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()

    for _ in range(num_epoch):
        for batch_idx, samples in enumerate(fold_dataloader):
            X_batch, y_batch = samples
            y_pred = model(X_batch)
            loss = loss_func(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}".format(
                _ + 1, num_epoch, batch_idx + 1, len(fold_dataloader), loss.item()
            ))

    with torch.no_grad():
        X_valid, y_valid = torch.from_numpy(X_valid), torch.from_numpy(y_valid)
        prediction = model(X_valid.float())
        error = ((prediction - y_valid) ** 2).mean()
        kfold_loss.append(error.item())
