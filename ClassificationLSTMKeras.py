#!/usr/bin/env python

from ROOT import TMVA, TFile, TTree, TCut
from subprocess import call
from os.path import isfile
import sys
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Reshape
from keras.regularizers import l2

# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

output = TFile.Open('TMVA_LSTM_PyKeras.root', 'RECREATE')
factory = TMVA.Factory('TMVAClassification', output,
                       '!V:!Silent:Color:DrawProgressBar:Transformations=None:AnalysisType=Classification')

############################Loading the data file
data = TFile.Open("/home/jui/Desktop/tmva/sample_timedata_t10_d30.root")
print(data.ls())
signal = data.Get('sgn')
background = data.Get('bkg')
# print(signal.Print())
# var = signal.GetBranch('vars_time0')
# sig = []
# for event in signal:
#     sig.append(getattr(event, 'vars_time0'))
#     break
# print(sig)
# print(sig.GetEntries())

# for event in bkg:
#     print(event.vars_time0)
#     break
# signal = np.asarray([[sig.vars_time0] for event in sig])
# # print(signal.shape)
# ###########################################################################
dataloader = TMVA.DataLoader('dataset_evaltest2')

for branch in signal.GetListOfBranches():
    dataloader.AddVariablesArray(branch.GetName(),30)

dataloader.AddSignalTree(signal, 1.0)
dataloader.AddBackgroundTree(background, 1.0)
dataloader.PrepareTrainingAndTestTree(TCut(''),
                                      'nTrain_Signal=8000:nTrain_Background=8000:SplitMode=Random:NormMode=NumEvents:!CalcCorrelations:!V')

# Generate model
n_steps = 10
n_features = 30
# Define model
model = Sequential()
model.add(Reshape((n_steps,n_features), input_shape=(300,)))
model.add(LSTM(75,activation='tanh', recurrent_activation='sigmoid', input_shape=(n_steps, n_features)))
model.add(Dense(2, activation='sigmoid'))

# Set loss and optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',])

# Store model to file
model.save('model_lstm_keras.h5')
model.summary()

# Book methods
factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras',
                   'H:!V:VarTransform=None:FilenameModel=model_lstm_keras.h5:NumEpochs=20:BatchSize=128')

# Run training, test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
