##############################LIBRARIES########################################################
from sys import exit
from time import gmtime, strftime

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as metrics

try:
    import numpy as np
except:
    print("Failed to import numpy.")
    exit()
try:
    import matplotlib.pyplot as plt
except:
    print("Failed to import matplotlib.")
    exit()
try:
    from sklearn.model_selection import train_test_split as tts
except:
    print("Failed to import from sklearn.")
    exit()
try:
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.autograd import Variable
    import torch.optim as optim
except:
    print("Cannot fetch/detect PyTorch.")
    exit()   
try:
    import pandas as pd
except:
    print("Failed to import Pandas.")
    exit()
#################################LOADING THE DATA##############################################
signal = np.load('./pytorch_signal.npy')
background = np.load('./pytorch_background.npy')


X1, X_test1, y1, y_test1 = tts(signal, np.zeros((10000,1)), test_size = 0.2)
X2, X_test2, y2, y_test2 = tts(background, np.ones((10000,1)), test_size = 0.2)

X_train = np.concatenate((X1, X2), axis=0)
y_train = np.concatenate((y1, y2), axis=0)

X_test = np.concatenate((X_test1, X_test2), axis=0)
y_test = np.concatenate((y_test1, y_test2), axis=0)
####################################DATALOADER#################################################
class dataset_class(Dataset):

    def __init__(self, x, y, train=True):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        # a = self.x[i:i+1000,:,:]
        return self.x[index,:], self.y[index,:]
    
    def __len__(self):
        return int((self.x.shape[0]))

trainset = dataset_class(X_train, y_train)
testset = dataset_class(X_test, y_test)

trainloader = DataLoader(dataset = trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(dataset = testset, batch_size=128, shuffle=False, num_workers=2)

sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
######################################CLASS####################################################
class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = lstm_layer = nn.LSTM(30, 75, 1, batch_first=True)
        self.dense = nn.Linear(750, 2)

    def init_hidden(self):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(1,128,75).to(next(self.parameters()).device)
        cell_state = torch.zeros(1,128,75).to(next(self.parameters()).device)
        self.hidden = (hidden_state.double(), cell_state.double())

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x,(self.hidden))
        # self.hidden = sigmoid(self.hidden)
        # self.hidden = (self.hidden)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        x = (lstm_out).contiguous().view(128,-1)
        x = sigmoid(self.dense(tanh(x)))
        return x
#################################INTIALIZERS##################################################
net = LSTM()
net = net.double()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
nepochs = 20
###################################TRAINING####################################################
isTrain = True

PATH = './pytorch_lstm.pth'

if(isTrain):
    net = net.train()
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    for epoch in range(nepochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
           
            inputs, labels = data

            inputs = inputs.reshape(inputs.shape[0],10,30)

            inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            net.init_hidden()
            outputs = net(inputs.double())
            # print(outputs.shape)
            # print(labels.shape)
            loss = criterion(outputs, labels[:,0].type(torch.long))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # if i % 1000 == 999:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.18f' %
            (epoch + 1, i + 1, running_loss /128))


    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    torch.save(net.state_dict(), PATH)
    print('Finished Training')
#####################################TESTING###################################################
isTest = True
if(isTest==True):
    net.load_state_dict(torch.load(PATH))
    predicted = np.array(([4]))

    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            if(i==31):
                break
            images, labels = data
            net.init_hidden()
            images = images.reshape(images.shape[0],10,30)
            images, labels = Variable(images), Variable(labels)
            
            outputs = net(images)
            _, pred_label=torch.max(outputs.data, dim=1)
            
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            if(i==31):
                break
            images, labels = data
            net.init_hidden()
            images = images.reshape(images.shape[0],10,30)
            images, labels = Variable(images), Variable(labels)
            
            outputs = net(images)
            # print(predicted.shape)
            _, pred_label =torch.max(outputs.data, dim=1)
            # print(np.array(pred_label).T.shape)
            predicted = np.concatenate((predicted, np.array(pred_label).T))
            
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    predicted = predicted[1:]
    ######################################METRICS##################################################
    from sklearn import metrics

    fpr, tpr, thresholds = metrics.roc_curve(y_test[:3968], predicted)
    plt.figure(1)
    plt.plot(fpr, tpr)
    plt.grid()
    plt.show()
    print('AUC score:',end = '')
    print(metrics.roc_auc_score(y_test[:3968], predicted))

    print('\n\nPrecision Recall F_score: ' , end='')
    print(metrics.precision_recall_fscore_support( y_test[:3968], predicted , average='binary'), end='\n')
    print('\n\nAccuracy: ', end ='')
    print(metrics.accuracy_score( y_test[:3968], predicted)*100,end ='%')
    # ###############################################################################################
