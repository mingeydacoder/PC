# %%
# ===========================================
# Homework 8 for 統計與大氣科學
# Prepared by Yi-Jhen Zeng at NTU, May 2022
# ===========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===========================================
# Define network & functions
# ===========================================
def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()

    return np

class LogisticRegression(nn.Module):

    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))

        return x

def train(model, x, y, optimizer):
    x, y = x.to(device), y.to(device)
    model.zero_grad()    # Set gradients of all model parameters to zero
    output = model(x)
    output = torch.squeeze(output)
    loss = F.binary_cross_entropy(output, y)
    loss.backward()
    optimizer.step()

    return loss, output

def test(model, x, y):
    model.eval()    # switch model to evaluate mode
    output = model(x)
    output = torch.squeeze(output)
    loss = F.binary_cross_entropy(output, y)

    return loss, output

def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    err_num = (y_true != predicted).sum()
    accuracy = (y_true == predicted).sum().float()/len(y_true)

    return err_num, accuracy

# ===========================================
# Data: zonal wind at 10hPa, 60 deg N
# ===========================================
# ------- Train & test data --------
#path = '/work/home/YCL.gajong/courses/statistics/logistic_regression/'
path = './'
dfx = pd.read_csv(path+'hw10_u_x.csv', skiprows=0, usecols=np.arange(1,145)) # cases(200)xlon(144)
dfy = pd.read_csv(path+'hw10_u_y.csv', skiprows=0, usecols=[1]) # 200x1
x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.3, random_state=42)

# convert all fields to Tensors
x_train = torch.from_numpy(x_train.to_numpy()).float()
y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())
x_test = torch.from_numpy(x_test.to_numpy()).float()
y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())
print('x_train:', x_train.shape, 'y_train:', y_train.shape)
print('x_test:', x_test.shape,'y_test:',  y_test.shape)

lon = np.linspace(0,358.75,144)

# ===========================================
# Train & Test the model
# ===========================================
# ------- model setting --------
input_size  = 144
output_size = 1
EPOCHS      = 60
lr          = 0.001  # learning rate
wd          = 0.1    # weight decay
model_logreg = LogisticRegression(input_size, output_size)
model_logreg.to(device)
# optimizer: stochastic gradient descent
optimizer = optim.SGD(model_logreg.parameters(), lr=lr, weight_decay=wd)

Accuracy = np.zeros((EPOCHS, 2))  # epochs, train/test
Loss = np.zeros((EPOCHS, 2))
for epoch in range(EPOCHS):
    x_train = x_train.view(-1, 144)
    x_train, y_train = x_train.to(device), y_train.to(device)

    train_loss, train_pred = train(model_logreg, x_train, y_train, optimizer)
    train_errnum, train_acc = calculate_accuracy(y_train, train_pred)

    test_loss, test_pred = test(model_logreg, x_test, y_test)
    test_errnum, test_acc = calculate_accuracy(y_test, test_pred)

    print('Epoch {}'.format(epoch+1))
    print('     Train set - loss: {}'.format(train_loss),' accuracy: {}'.format(train_acc))
    print('     Test set - loss: {}'.format(test_loss),' accuracy: {}'.format(test_acc))
    print('     Number of test errors: {}'.format(test_errnum))

    Accuracy[epoch,0], Accuracy[epoch,1] = train_acc, test_acc
    Loss[epoch,0], Loss[epoch,1] = train_loss, test_loss

# %%

# ===========================================
# plot
# ===========================================
plt.close('all')
fig = plt.figure(figsize=(8,8))
ax0 = fig.add_subplot(211)
ax1 = fig.add_subplot(212)
ax0.plot(np.arange(0,EPOCHS), Accuracy[:,0], label='train')
ax0.plot(np.arange(0,EPOCHS), Accuracy[:,1], label='test')
#ax0.set_xscale('log')
ax0.legend()
ax0.title.set_text('accuracy')
ax1.plot(np.arange(0,EPOCHS), Loss[:,0], label='train')
ax1.plot(np.arange(0,EPOCHS), Loss[:,1], label='test')
#ax1.set_xscale('log')
ax1.legend()
ax1.title.set_text('loss')
ax1.set_xlabel('epochs')
# %%
plt.show()
