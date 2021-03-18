import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random, time
from utils.loader import stock_loaders
from os import path

DATA_FILE = path.join('data', 'split_categorical.csv')
DROP_NUM = 0
batch_size = 64
test_batch_size = 64

train_loader, test_loader = stock_loaders(batch_size, DATA_FILE, False, DROP_NUM)

# The number of epochs is at least 10, you can increase it to achieve better performance
num_epochs = 20
learning_rate = 1e-3

print("Starting training for fully-connected with ReLU")
# Define the model for fully-connected with ReLU
class LinNet(nn.Module):
    def __init__(self):
        super(LinNet, self).__init__()
        self.fc1 = nn.Linear(17 - DROP_NUM, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x   

model = LinNet()

# For training on GPU
# model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,  momentum=0.9)
criterion = nn.CrossEntropyLoss()
epoch_num = 1

# Training the Model
for epoch in range(num_epochs):
    total_loss = 0
    for i, (images, labels, _) in enumerate(train_loader):
        # For training on GPU
        # images = images.cuda()
        # labels = labels.cuda()

        # Compute predicted value
        y_pred = model(images.float())
        
        # Compute loss
        loss = criterion(y_pred, labels.long())
        total_loss += loss.item()

        # Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ## Print your results every epoch
    print("Epoch {} Avg Loss: {}".format(epoch_num, total_loss/len(train_loader)))
    epoch_num += 1

# Test the Model
f = open("predictions_3class_categorical.txt", "w")
correct = 0.
total = 0.
for images, labels, stocks in test_loader:
    # For training on GPU
    # images = images.cuda()
    # labels = labels.cuda()

    outputs = model(images.float()).squeeze(1)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    
    for ind, stock in enumerate(stocks):
        f.write('{}: {}, {}\n'.format(stock, predicted[ind], labels[ind]))

print('Accuracy of the model on the test images: %f %%' % (100 * (correct / total)))