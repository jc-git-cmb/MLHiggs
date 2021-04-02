import numpy as np 
import math 
import pandas as pd 
import os
import h5py
from matplotlib import pyplot as plt 
from pylorentz import Momentum4

import tensorflow as tf
import tensorflow.keras as keras 
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Layer
from sklearn.metrics import roc_curve

from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras import callbacks 


#import functions for ANN
import importlib
import SHP_FuncANN
import SHP_Func2

importlib.reload(SHP_FuncANN)
importlib.reload(SHP_Func2)

#input random seed - reproducability of NN = neural network 
np.random.seed(7)

#input dataset and determine 1/2 length and 3/4 length (of number of rows/events)
dataset = pd.read_csv( 'dataset_new_nn1.csv', header = 0 , delimiter = "," )
inv_mass_dframe = pd.read_csv( 'inv_mass_full_data.csv', header = 0 , delimiter = "," )
#dataset = pd.read_csv( 'comparison_data2.csv', header = 0 , delimiter = "," )

half_data = np.int( 0.5 * len(dataset) )
three_quarters_data = np.int( 0.75 * len(dataset) )

''' Define input Parameters'''
train_X = dataset.iloc[0:half_data, 1:34 ]
train_Y = dataset.iloc[0:half_data, 0 ]

#assign half the dataset to testing the model, test_X = inputs, test_Y = known outputs 
test_X = dataset.iloc[ half_data:-1, 1:34  ]
test_Y = dataset.iloc[ half_data:-1, 0 ]

#convert the training and testing inputs to have a mean approx = 0 and standard deviation = 1 
#this was done for optimal performance of the network
train_X_norm_std = StandardScaler().fit_transform(train_X)
test_X_norm_std = StandardScaler().fit_transform(test_X)

M_train = inv_mass_dframe[0:half_data]
M_test = inv_mass_dframe[half_data:-1]

#scale the inputs between 0 and 1
m_train  = M_train - M_train.min()
m_train /= M_train.max()

m_test1 = M_test - M_test.min()
m_test = m_test1 / M_test.max()

print('DONE DATA MANIPULATION')

''' Classifier Neural Network '''
#define the classifier network model  as before 
def classifier_nn(input_dim):

    # Inputs - X_train(input_dim = number columns)
    inputs = Input(shape=(input_dim,))
    
    # Hidden layers
    x = Dense(300, activation='relu')(inputs)
    x = Dense(200, activation='relu')(x)
    
    # Outputs - 0,1 binary 
    outputs = Dense(1, activation='sigmoid')(x)    
    
    # Return the model for the classifier neural network
    return Model( inputs = inputs , outputs = outputs , name = 'classifier' )

#call the early stopping call back on Keras
earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  mode ="min", patience = 20, restore_best_weights = True) 
 
# Contruct classifier network
clf = classifier_nn(33)
clf.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
# Model summary
clf.summary()
#fit the classifier 
clf_nn = clf.fit( train_X_norm_std, train_Y, epochs = 100, batch_size = 5000, validation_split=0.2, callbacks =[earlystopping], class_weight = {0:1, 1:4.7})


#obtain the predicted outputs from the fitted classifier using the training data set - for adversary during testing of model
pred = clf.predict(train_X_norm_std)

#plot the loss function during training
probs = clf.predict(test_X_norm_std)
plt.plot(clf_nn.history['loss'], label = 'train')
plt.plot(clf_nn.history['val_loss'], label = 'validation')
plt.title('Loss function value during training of Classifier ')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.annotate('ATLAS Work in Progress', xy = (70, 0.75))
plt.legend()
plt.show()

#plot ROC curve
probabilities2 = clf.predict(test_X_norm_std)
print('AUC : ' + str(roc_auc_score(test_Y, probabilities)))
fp, tp, threshold = roc_curve(test_Y, probabilities)
plt.xlabel('false positive rate', fontsize=12)
plt.ylabel('true positive rate', fontsize=12)
plt.title( 'ROC - Receiver Operating Characterisitics : 50/50 train-test' )
plt.plot(fp,tp, color = 'y')
plt.plot([0, 1], [0, 1], linestyle = '--', color = 'b')
plt.annotate('ATLAS Work in Progress', xy = (0.7, 0.))
plt.show()

''' Adversary Network '''
def adversary_nn(n):

    # Input - initialises a Keras tensor
    input_1 = Input(shape=(1,))
    input_2 = Input(shape=(1,))
    
    # Hidden layers - Dense creates a densely-connected NN layer
    x = Dense(200,  activation='relu')(input_1)
    x = Dense(200,  activation='relu')(x)
    x = Dense(200, activation='relu')(x)

    # Gaussian mixture model (GMM) components
    coeffs = Dense(n, activation='softmax')(x)  # GMM coefficients sum to one
    means  = Dense(n, activation='sigmoid')(x)  # Means are on [0, 1]
    sigma = Dense(n, activation='softplus')(x)  # Widths are positive 
    
    # Posterior probability density function - outputs of ANN 
    pdf = SHP_FuncANN.PosteriorLayer(n)([coeffs, means, sigma, input_2])

    # Build model
    return Model(inputs=[input_1 , input_2], outputs=pdf, name='adversary')

#loss function for adversary 
def loss_ANN(y_true, y_pred):
    loss_adv = -K.log(y_pred)
    return loss_adv

#call adversary
adv = adversary_nn(5)
#compile the adversary network
adv.compile(optimizer = 'adam', loss=loss_ANN )
#print a summary 
adv.summary()
#fit the adversary network
ann = adv.fit([pred, m_train], np.ones_like(m_train), epochs = 20 , validation_split=0.2, batch_size = 5000, callbacks =[earlystopping], class_weight = {0:1, 1:4.7})

#plot the loss function during training of the Adversary
plt.plot(ann.history['loss'], label = 'train')
plt.plot(ann.history['val_loss'], label = 'validation')
plt.title('Loss function value during training of Adversarial neural networks')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.text( 1 , 1 , 'ATLAS work in Progress',  size=10)
plt.legend()
plt.show()


''' Combined Neural Network '''

def combined_model (classifier, adversary, lambda_val, learn_ratio):

    # Classifier inputs and outputs, get input shape from the first layer of the model (input dim)
    # input_m - input mass to ANN
    input_clf  = Input( shape = classifier.layers[0].input_shape[0][1] )
    input_mass    = Input( shape=(1,) )
    output_clf = classifier( input_clf )

    # Connect the network with gradient reversal layer from the SHP_FuncANN file
    gradient_reversal = SHP_FuncANN.GradientReversalLayer(lambda_val * learn_ratio)(output_clf)
    
    # Call Adversary NN
    output_adv = adversary([gradient_reversal, input_mass])

    # Build combined model with given inputs and outputs for both the adversary and classifier nn's
    return Model(inputs=[input_clf, input_mass], outputs=[output_clf, output_adv], name='combined')


#loss function weighting for classifier and adversary respectively and Lambda - Input at terminal
loss_weights = [float(input("Learning Ratio: ")) , 1. ]
lambda_val = float(input("Lambda: ")) 

#new loss function for combined model adversary * lambda
def loss_ANN2(y_true, y_pred):
    loss_adv = -K.log(y_pred)*lambda_val
    return loss_adv

#call on classifier and adversary functions
adversay = adversary_nn(5)
classifier = classifier_nn(33)

#call on combined network functions, compile it and print the summary of the model
combined = combined_model(classifier , adversay, lambda_val = lambda_val, learn_ratio = loss_weights[0] / loss_weights[1] )
combined.compile( loss = ['binary_crossentropy', loss_ANN2] , optimizer = 'adam' , loss_weights = loss_weights)
combined.summary()

# Prepare sample weights (i.e. only do mass-decorrelation for background)
y_val = ((train_Y == 0).astype(float)).to_numpy()
weight_1 = np.ones(shape=(len(train_Y),))
weight_2 = y_val * np.sum(weight_1) / np.sum(y_val)

#prepare class weights - more background than signal events 
class_weights = [{0:1, 1:4.7}, {0:1, 1:4.7}]

#fit the combned model
cmb_nn = combined.fit([train_X_norm_std, m_train], [train_Y, np.ones_like(m_train)], sample_weight = [weight_1, weight_2], class_weight = class_weights, epochs = 100, batch_size = 5000, validation_split = 0.2, callbacks =[earlystopping])

#plot the loss function during training 
plt.plot(cmb_nn.history['loss'], label = 'train')
plt.plot(cmb_nn.history['val_loss'], label = 'validation')
plt.title('Loss function value during training of Combined Classifier and Adversarial neural networks')
plt.xlabel('Epoch' , fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.text( 1 , 1 , 'ATLAS work in Progress',  size=10)
plt.legend()
plt.show()

#predict the new classifier predictions after training of the combined model
probabilities2 = classifier.predict(test_X_norm_std)

#plot the ROC curve
print('AUC : ' + str(roc_auc_score(test_Y, probabilities2)))
fp, tp, threshold = roc_curve(test_Y, probabilities2)
plt.xlabel('false positive rate', fontsize=20)
plt.ylabel('true positive rate', fontsize=20)
plt.title( 'ROC - Receiver Operating Characterisitics' , fontsize=20)
plt.annotate('ATLAS Work in Progress', xy = (0.6, 0), fontsize = 16)
plt.plot(fp,tp, color = 'y')
plt.plot([0, 1], [0, 1], linestyle = '--', color = 'b')
plt.show()


