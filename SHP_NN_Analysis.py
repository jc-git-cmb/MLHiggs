import numpy as np 
import math 
import pandas as pd 
import tensorflow as tf
from matplotlib import pyplot as plt 
from pylorentz import Momentum4
import tensorflow.keras as keras 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import roc_curve
from keras.backend import clear_session
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import os
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import model_from_json

#DATA MANIPULATION AND SAVE FOR USE IN OTHER MODELS

dataset = pd.read_csv( 'dataset_new_nn1.csv', header= 0 , delimiter = "," )
half_data = np.int( 0.5 * len(dataset) )
three_quarters_data = np.int( 0.75 * len(dataset) )

##Model 1 - all columns in dataset - up to first jet

#assign half the dataset to traininhg data, train_X = inputs, train_Y = outputs 
train_X = dataset.iloc[0:three_quarters_data, 1:14 ]
train_Y = dataset.iloc[0:three_quarters_data, 0 ]

#assign half the dataset to testing the model, test_X = inputs, test_Y = known outputs 
test_X = dataset.iloc[three_quarters_data:-1, 1:14  ]
test_Y = dataset.iloc[three_quarters_data:-1, 0 ]

test_Xdframe = test_X.reset_index(drop=True)
test_Ydframe = test_Y.reset_index(drop=True)

print(test_Ydframe)

#define invariant mass values variables 
y1_eta = dataset.iloc[ :, 2 ]
y1_phi = dataset.iloc[ :, 3]
y1_pt = dataset.iloc[ :, 1 ]

y2_eta2 = dataset.iloc[ :, 6 ]
y2_phi2 = dataset.iloc[ :, 7]
y2_pt2 = dataset.iloc[ :, 5]

#create array of zeros - mass of photons is 0  
zeros = np.zeros( len(dataset) )
#define the 4-momentum vectors for each photon using pylorentz
y1 = Momentum4.m_eta_phi_pt(  zeros , y1_eta , y1_phi , y1_pt )
y2 = Momentum4.m_eta_phi_pt( zeros , y2_eta2 , y2_phi2 , y2_pt2 )

#sum the 4-vectors of the two photons 
parent = y1 + y2 

#find the invariant mass of the two photons 
inv_mass = parent.m 

#convert inv_mass array to a pandas DataFrame 
inv_mass_dframe = pd.DataFrame( inv_mass ) 



probabilities = np.loadtxt('probabilities_nn1.txt')
prob_0 = []

for i in range(len(probabilities)):
    prob_0.append(1-probabilities[i])

prob_0_rev = prob_0[::-1]

plt.hist( probabilities, bins = 100, histtype = 'step', color = 'r', label = 'Signal' )
plt.hist( prob_0, bins = 100, histtype = 'step', color = 'y', label = 'Background' )
plt.title( 'ATLAS work in Progress : Probabilities of Signal and Background - test data' )
plt.xlabel( 'Probability' )
plt.yscale('log')
plt.legend()
plt.show()

"""
#define test variables 
test_eta = test_X.iloc[ :, 1 ]
test_phi = test_X.iloc[ :, 2]
test_pt = test_X.iloc[ :, 0 ]

test_eta2 = test_X.iloc[ :, 5 ]
test_phi2 = test_X.iloc[ :, 6]
test_pt2 = test_X.iloc[ :, 4 ]


#create array of zeros - mass of photons is 0  
zeros = np.zeros( len(test_Y) )
#define the 4-momentum vectors for each photon using pylorentz
y1 = Momentum4.m_eta_phi_pt(  zeros , test_eta , test_phi , test_pt )
y2 = Momentum4.m_eta_phi_pt( zeros , test_eta2 , test_phi2 , test_pt2 )

#sum the 4-vectors of the two photons 
parent_test = y1 + y2 

#find the invariant mass of the two photons 
inv_mass_test = parent_test.m 
prob_dframe = pd.DataFrame( probabilities )

inv_mass_test_dframe = pd.DataFrame( inv_mass_test )
"""

prob_dframe = pd.DataFrame( probabilities )
inv_mass = inv_mass_dframe.iloc[three_quarters_data:-1]
inv_mass_test_dframe = inv_mass.reset_index(drop=True)

sculpt_data1 = pd.concat( [ prob_dframe , test_Ydframe], axis = 1,  ignore_index=True)
sculpt_data2 = pd.concat( [ sculpt_data1 , test_Xdframe], axis = 1, ignore_index=True)
sculpt_data = pd.concat( [ sculpt_data2, inv_mass_test_dframe], axis = 1, ignore_index=True)

myy_bkg_sculpt = []

print(sculpt_data)

for i in range( len(test_Y) ):
    if sculpt_data.iloc[i, 0] > 0.7 and sculpt_data.iloc[i,1] == 0 :
        myy_bkg_sculpt.append(sculpt_data.iloc[i, 15])

plt.hist(myy_bkg_sculpt, histtype = 'step', bins = 600)
plt.title('ATLAS work in Progress : Sculpting of background data - invariant mass')
plt.xlabel('Invariant mass [MeV]')
plt.ylabel('Events')
plt.show()

