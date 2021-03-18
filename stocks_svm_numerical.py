import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random, time
import csv
from utils.loader import stock_loaders

DATA_FILE = 'split_numerical.csv'
DROP_NUM = 0
batch_size = 64
test_batch_size = 64

train_loader, test_loader = stock_loaders(batch_size, DATA_FILE, True, DROP_NUM)

# The number of epochs is at least 10, you can increase it to achieve better performance
num_epochs = 10
learning_rate = 0.01

print("Starting training for Linear SVM")
# Define the model for fully-connected with ReLU
class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.svm = nn.Linear(9 - DROP_NUM, 1)

    def forward(self, x):
        return self.svm(x)   

model = SVM()

# For training on GPU
# model = model.cuda()

def my_hinge_loss(output, target):
    loss = 1 - torch.multiply(output,target)
    loss = torch.clamp(loss,min=0.0,max=float('inf'))
    return torch.mean(loss)


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.HingeEmbeddingLoss()
epoch_num = 1

# Training the Model
for epoch in range(num_epochs):
    total_loss = 0
    for i, (images, labels, _) in enumerate(train_loader):
        # For training on GPU
        # images = images.cuda()
        # labels = labels.cuda()
        labels = (2*(labels.float()-0.5))
        # Compute predicted value
        y_pred = model(images.float()).squeeze(1)
        
        # Compute loss
        loss = criterion(y_pred, labels)
        total_loss += loss.item()

        # Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ## Print your results every epoch
    print("Epoch {} Avg Loss: {}".format(epoch_num, total_loss/len(train_loader)))
    epoch_num += 1

# Test the Model
f = open("predictions_2class_svm_numerical.txt", "w")
correct = 0.
total = 0.
for images, labels, stocks in test_loader:
    # For training on GPU
    # images = images.cuda()
    # labels = labels.cuda()

    outputs = model(images.float()).squeeze(1)
    predicted = outputs.data >= 0

    predicted = predicted.type_as(labels)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    for ind, stock in enumerate(stocks):
        f.write('{}: {}, {}\n'.format(stock, predicted[ind], labels[ind]))
    
print('Accuracy of the model on the test images: %f %%' % (100 * (correct / total)))