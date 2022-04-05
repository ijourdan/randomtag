#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 17:49:39 2022

@author: ivan
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import Data_Gen
from utils import Net
from utils import corpus_generator
from utils import shuffle_tags
from utils import plt_data_2d


#%%
# PARAMETERS
# Corpus
quant_tag = 4
quant_data = 100
dimension = 2
test_proportion = 0.2
gap=6
learning_rate = 0.001

# Shuffled tags
shuffle_probability = 0.9

# Training
num_epochs = 250
batch_size = 5

# =================================

# Data generation
training, training_tags, test, tags = corpus_generator(quant_tag, quant_data, dimension, test_proportion, gap)


new_ttags = shuffle_tags(training_tags, proba=shuffle_probability)

real_train_set = Data_Gen(training, training_tags)
shuffled_train_set = Data_Gen(training,new_ttags)
test_set = Data_Gen(test,tags)

train_loader = torch.utils.data.DataLoader(shuffled_train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, shuffle=False)

plt_data_2d(training, training_tags)
plt_data_2d(training,new_ttags)

#%%
#Training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Net(num_classes=quant_tag, num_input=dimension).to(device)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%%
# Entrenamiento del modelo
loss_out = []
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (points, labels) in enumerate(train_loader):
        points = points.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(points)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            loss_out.append(loss.item())

#%%
# Test the model
net_output = []
real_label = []
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for points, labels in test_loader:
        points = points.to(device)
        real_label.append(labels)
        outputs = model(points)
        _, predicted = torch.max(outputs.data, 1)
        net_output.append(predicted.to('cpu'))
        #total += labels.size(0)
        #correct += (predicted == labels).sum().item()
    net_output = torch.Tensor(net_output).long()
    real_label = torch.Tensor(real_label).long()
    #print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

plt_data_2d(test,net_output)
plt.show()
#%%


