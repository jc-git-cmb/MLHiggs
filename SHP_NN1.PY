import numpy as np 
import math 
import pandas as pd 
import tensorflow as tf
from matplotlib import pyplot as plt 
from pylorentz import Momentum4
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.backend import clear_session
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import os
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

''' Method 1 - Classifier Network '''

#random seed for reproducibility
np.random.seed(7)

#load data sets 
dataset = pd.read_csv( 'dataset_new_nn1.csv', header = 0 , delimiter = "," )
half_data = np.int( 0.5 * len(dataset) )
three_quarters_data = np.int( 0.75 * len(dataset) )

'''Inputs to network'''
#use 5 leading jets and 2 photons
#assign half the dataset to training data, train_X = inputs, train_Y = outputs 
train_X = dataset.iloc[0:half_data, 1:34 ]
train_Y = dataset.iloc[0:half_data, 0 ]

#assign half the dataset to testing the model, test_X = inputs, test_Y = known outputs 
test_X = dataset.iloc[ half_data:-1, 1:34  ]
test_Y = dataset.iloc[ half_data:-1, 0 ]
print(test_X)
print(test_Y)

#Transform the inputs to have a mean of 0 and standard deviation of approx 1
train_X_norm_std = StandardScaler().fit_transform(train_X)
test_X_norm_std = StandardScaler().fit_transform(test_X)
print('mean',train_X_norm_std.mean(axis=0)) # add this to check - close to zero but not exactly 
print('std',train_X_norm_std.std(axis=0))  # add this to check - returns 1 

print('mean',test_X_norm_std.mean(axis=0)) # add this to check - close to zero but not exactly 
print('std',test_X_norm_std.std(axis=0))  # add this to check - returns 1 


''' Classifier '''
def nn1():
    # define the keras model
    model = Sequential()
    #model.add(Dense( 300, input_dim = 33, activation='relu'))
    model.add(Dense( 300, input_dim = 33, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    #batch size - how many rows from dataset are considered before weights are updated 
    #epochs - number of iterations for training 
    model.compile( loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'] )

    return model 

model = nn1()

# fit the keras model on the dataset
nn1 = model.fit( train_X_norm_std, train_Y , epochs = 10, batch_size = 10000, validation_data = (test_X_norm_std, test_Y))

# evaluate the keras model
print(nn1.history.keys())


loss, accuracy = model.evaluate(train_X_norm_std, train_Y, verbose = 0)
print('\nLoss: %.2f, Accuracy: %.2f' % (loss, accuracy*100))

loss_test, accuracy_test = model.evaluate( test_X_norm_std, test_Y, verbose = 0 )
print('\nLoss: %.2f,Accuracy: %.2f' % (loss_test, accuracy_test*100))

#plot loss function during training 
print(nn1.history.keys())
plt.plot(nn1.history['loss'], label = 'train')
plt.plot(nn1.history['val_loss'], label = 'validation')
plt.xlabel('epoch')
plt.ylabel('loss function value')
plt.title('Loss Function during training')
plt.legend()
plt.show()

#plot accuracy during training 
plt.plot(nn1.history['accuracy'], label='train')
plt.plot(nn1.history['val_accuracy'], label='validation')
plt.xlabel('epoch')
plt.ylabel('accuracy value')
plt.title('Accuracy during training')
plt.legend()
plt.show()

''' ROC curve'''
#determine the probability of an event being a signal from the model
probabilities = model.predict( test_X_norm_std, batch_size = 1 )

#determinine the AUC score for the model given the true outputs and the probabilities determined from the model
print('AUC : ' + str(roc_auc_score(test_Y, probabilities)))

#determine the roc curve (fp = false positive, tp = true positive) and plot the false positive rate versus the true positive rate
fp, tp, threshold = roc_curve(test_Y, probabilities)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title( 'ROC - Receiver Operating Characterisitics : 50/50 train-test' )
plt.plot(fp,tp, color = 'y')
plt.plot([0, 1], [0, 1], linestyle = '--', color = 'b')
plt.show()

''' VARIABLE RANKING METHOD - Error in matching importance to correct variable, did not have time to fix '''
"""
from sklearn.inspection import permutation_importance

#variables being compared
col_yy1 = ['y1_pt', 'y1_eta' , 'y1_phi' , 'y1_E' , 'y2_pt' , 'y2_eta' , 'y2_phi' , 'y2_E', 'jet1_pt' , 'jet1_eta', 'jet1_phi' , 'jet1_E' , 'jet1_DLR1', 'jet2_pt' , 'jet2_eta' , 'jet2_phi' ,'jet2_E' , 'jet2_DLR1', 'jet3_pt','jet3_eta','jet3_phi','jet3_E','jet3_DLR1', 'jet4_pt','jet4_eta','jet4_phi','jet4_E','jet4_DLR1', 'jet5_pt','jet5_eta','jet5_phi','jet5_E','jet5_DLR1']

#use a scikit-leran classifier wrapperand fit the function
mod_wrapper = KerasClassifier( build_fn = nn1, epochs=82, batch_size=5000 )
print('done')
mod_wrapper.fit( train_X_norm_std, train_Y )
print('done')

#perform permutation importance 
r = permutation_importance(mod_wrapper , test_X_norm_std[0:100000, :], test_Y.iloc[0:100000], n_repeats = 5, random_state=0)

#loop through importances and assign mean value to the corresponding variable - error in this
importances = []
for i in r.importances_mean.argsort()[::-1]:
    print(f"{col_yy1[i]:<8}" f"{r.importances_mean[i]:.3f}"f" +/- {r.importances_std[i]:.3f}")
    importances.append(r.importances_mean[i])

print('done')
x = []
for j in range(0, len(col_yy1)):
    x.append(j+1)
print(len(importances))
print(len(x))
plt.barh(x, width = importances, tick_label = col_yy1)
plt.show()

"""









