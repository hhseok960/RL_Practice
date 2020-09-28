import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN_binary(nn.Module):
    def __init__(self):
        # image input size : 150 X 150 X 3
        super(CNN_binary, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.lin1 = nn.Linear(128*7*7, 512)
        self.lin2 = nn.Linear(512, 2)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128*7*7)
        x = F.relu(self.lin1(x))
        x = self.out_act(self.lin2(x))
        return x


"""
from torchsummary import summary
model = CNN_binary().to(device)
summary(model, (3, 150, 150))
"""

### Dataset Prepare - dog:0 / cat:1
def read_image_dataset(path, xlen=150, ylen=150):
    input, target = [], []
    for i in os.listdir(path):
        file_path = path + "/{0}".format(i)
        image = cv2.imread(file_path)
        image = cv2.resize(image, dsize=(xlen, ylen))
        image = np.asarray(image, dtype=np.float32)
        image /= 255
        input.append(image)
        if i[:3] == "cat":
            target.append(1)
        elif i[:3] == "dog":
            target.append(0)
    input, target = np.array(input), np.array(target)
    return input, target


path = "dogs-vs-cats"
X_train, y_train = read_image_dataset(path + "/train")
X_valid, y_valid = read_image_dataset(path + "/validation")
X_test, y_test = read_image_dataset(path + "/test")

X_train = np.transpose(X_train, (0, 3, 1, 2))
X_valid = np.transpose(X_valid, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))
### Dataset Prepare Complete

num_epoch = 30
batch_size = 128
learning_rate = 1e-3

train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
X_valid, y_valid = torch.from_numpy(X_valid).float(), torch.from_numpy(y_valid)
X_valid, y_valid = X_valid.to(device), y_valid.to(device)

model = CNN_binary().to(device)
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()
loss_arr = []
valid_loss = []

for i in range(num_epoch):
    loss_log = []
    for batch_idx, samples in enumerate(dataloader):
        X_batch, y_batch = samples
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch)
        loss = loss_func(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        loss_log.append(loss.cpu().detach().numpy())
        print("Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}".format(
            i + 1, num_epoch, batch_idx + 1, len(dataloader), loss.item()
        ))
    loss_arr.append(loss_log)
    # with torch.no_grad():
    #     y_hat = model(X_valid)
    #     correct_prediction = torch.argmax(y_hat, 1) == y_valid
    #     accuracy = correct_prediction.float().mean()
    #     valid_loss.append(accuracy.cpu().detach().numpy())


