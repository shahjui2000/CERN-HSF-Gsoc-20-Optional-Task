#!/usr/bin/env python
##############################LIBRARIES########################################################
from ROOT import TFile
from sys import exit
from time import gmtime, strftime
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
    import pandas as pd
except:
    print("Failed to import Pandas.")
    exit()
#################################LOADING THE DATA##############################################
data = TFile.Open("/home/jui/Desktop/tmva/sample_timedata_t10_d30.root")
print(data.ls())

sig = data.Get('sgn')
# bkg = data.Get('bkg')
i = 0
print('Ola')


signal = np.zeros((1,300))
for event in sig:
    tmp = np.zeros((1,30))
    tmp = np.hstack((tmp, np.vstack(sig.vars_time0).T)) 
    tmp = np.hstack((tmp, np.vstack(sig.vars_time1).T)) 
    tmp = np.hstack((tmp, np.vstack(sig.vars_time2).T)) 
    tmp = np.hstack((tmp, np.vstack(sig.vars_time3).T)) 
    tmp = np.hstack((tmp, np.vstack(sig.vars_time4).T))  
    tmp = np.hstack((tmp, np.vstack(sig.vars_time5).T))  
    tmp = np.hstack((tmp, np.vstack(sig.vars_time6).T)) 
    tmp = np.hstack((tmp, np.vstack(sig.vars_time7).T))  
    tmp = np.hstack((tmp, np.vstack(sig.vars_time8).T))  
    tmp = np.hstack((tmp, np.vstack(sig.vars_time9).T))      
    signal = np.vstack((signal, tmp[:,30:] ))       
    # break
signal = signal[1:,:]
print(signal.shape)


# background = np.zeros((1,30))
# for event in bkg:
#     background = np.vstack((background, np.vstack(bkg.vars).T))
# background = background[1:,:]
# print(background.shape)

np.save('pytorch_signal', signal)