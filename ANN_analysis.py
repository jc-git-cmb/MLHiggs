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
import scipy 
from scipy.optimize import curve_fit
from scipy import special 

''' Probabilities for lam = 50, lerning ratio = 1e-5 - Combined'''
#probabilities = np.loadtxt('probabilities_nn1_50_50_5jets_lam50_lr5_clw.txt')

''' Probabilities for Comparison'''
#probabilities = np.loadtxt('probabilities_clf_50_50_5jets_comparison_clw.txt')

''' Probabilities for Classifier'''
#probabilities = np.loadtxt('probabilities_clf_50_50_5jets_final')


''' Probabilities for lam = 5, lerning ratio = 1e-5 - Combined'''
#probabilities = np.loadtxt('probabilities_nn1_50_50_5jets_lam5_lr5_clw_final.txt')

'''Read in files using pandas '''
dataset = pd.read_csv( 'dataset_new_nn1.csv', header = 0 , delimiter = "," )
inv_mass = pd.read_csv( 'inv_mass_full_data.csv', header = 0 , delimiter = "," )

half_data = np.int( 0.5 * len(dataset) )
inv_mass_dframe = inv_mass.iloc[half_data:-1, 0 ]


'''Make the combine DataFrame'''
inv_m = inv_mass_dframe.reset_index(drop=True)
prob = pd.DataFrame( probabilities )
prob_df = prob.reset_index(drop=True)
dat1 = dataset.iloc[half_data:-1, : ]
dat2 = dat1.reset_index(drop=True)
dat3 = pd.concat( [prob , inv_m], axis = 1)
data = pd.concat( [dat3 , dat2] , axis = 1)


'''Discriminant'''

prob_sig = []
prob_bkg = []

for i in range( 0, len(probabilities) ): 
    #find probabilities of signal events 
    if data.iloc[i, 2] == 1:
        prob_sig.append( probabilities[i] )
    #find probabilities of background events 
    elif data.iloc[i, 2] == 0:
        prob_bkg.append( probabilities[i] ) 


#define a weight function for histogram scaling 
def weights(m):
    sum_weights = float( len(m) )
    numerators = np.ones_like( m )
    return( numerators / sum_weights )

#number of bns for discriminant 
def bin(data): 
    bin = np.max(data) * 100
    return np.int(bin)

#Histogram - Discriminant 
plt.hist( prob_sig, bins = bin(prob_sig), histtype = 'step', color = 'r', label = 'Signal', weights = weights(prob_sig))
plt.hist( prob_bkg, bins = bin(prob_bkg), histtype = 'step', color = 'y', label = 'Background', weights = weights(prob_bkg))
plt.title( 'Probabilities of Signal and Background - test data', fontsize=20)
plt.xlabel( 'Probability' , fontsize=20)
plt.ylabel( 'Fraction of events per 1GeV', fontsize=20)
plt.annotate('ATLAS Work in Progress', xy = (0.7, 0.2), fontsize=16)
plt.xticks(fontsize = 16 )
plt.yticks(fontsize = 16 )
plt.legend(fontsize = 16)
plt.show()

''' SENSITIVITY and SCULPTING'''
#signal efficiency 
#sf sig = 8.1330E-5 
#sf bkg = 0.00189

def efficiency():

    sig = []
    bkg = []
    bkg_full = []

    for i in range( 0 , len(data) ):
        #find mass of signal events with probability > 0.7
        if data.iloc[i, 0] > 0.7 and data.iloc[i, 2] == 1 :
            sig.append( data.iloc[i, 1] )
        #find mass of background events with probability > 0.7
        if data.iloc[i, 0] > 0.7 and data.iloc[i, 2] == 0 :
            bkg.append( data.iloc[i, 1])
        #find mass of background events with probability > 0
        if data.iloc[i, 0] > 0 and data.iloc[i, 2] == 0 :
            bkg_full.append( data.iloc[i, 1])

    #cut these lists so the mass is cnfined to the ranges specified
    sig_cut = []
    bkg_cut = []
    sig_small_cut = []
    bkg_full_cut = []
    
    for j in range(0, len(sig)):
        if sig[j] > 105000 and sig[j] < 160000: 
            sig_cut.append(sig[j])
        if sig[j] > 121000 and sig[j] < 129000: 
            sig_small_cut.append(sig[j])

    for k in range(0, len(bkg)):
        if bkg[k] > 105000 and bkg[k] < 160000: 
            bkg_cut.append(bkg[k])
    
    for m in range(0, len(bkg_full)):
        if bkg_full[m] > 105000 and bkg_full[m] < 160000: 
            bkg_full_cut.append(bkg_full[m])
    
    return sig_cut, bkg_cut, sig_small_cut, bkg_full_cut

print('done')

eff_data = efficiency()
sig = eff_data[0]
bkg = eff_data[1]
sig_small = eff_data[2]
bkg_full_cut = eff_data[3]

#number of bins for invariant mass plots
def bin(data): 
    bin = (np.max(data) - np.min(data)) / 1000
    return np.int(bin)

'''SCULPTING - plot bkg cut and full bkg'''
counts_bkg1, bins_bkg1, p_bkg1 = plt.hist(bkg_full_cut, histtype = 'step', bins = bin(bkg_full_cut), weights = weights(bkg_full_cut), label = 'Classifier, full background')
counts_bkg2, bins_bkg2, p_bkg2 = plt.hist(bkg, histtype = 'step', bins = bin(bkg), weights = weights(bkg), label = 'Classifier, prob>0.7', color = 'r')
#plt.annotate('ATLAS Work in Progress', xy = (145000, 0.024), fontsize=16)
#plt.annotate('ATLAS Work in Progress', xy = (145000, 0.03), fontsize=16)
plt.annotate('ATLAS Work in Progress', xy = (145000, 0.09), fontsize=20)
plt.errorbar(bins_bkg2[0:54]+ 0.5 * (bins_bkg2[1] - bins_bkg2[0]), counts_bkg2[0:55], yerr = np.sqrt(counts_bkg2[0:54] * weights(bkg)[0]), fmt='_r', ms = (0.1))
plt.xlabel('Invariant mass [MeV]', fontsize=20)
plt.ylabel('Fraction of Events per 1GeV', fontsize=20)
plt.title('Sculpting of background data - invariant mass', fontsize=20)
plt.xticks(fontsize = 20 )
plt.yticks(fontsize = 20 )
plt.legend(fontsize = 20)
plt.show()


'''NN/N_bkg_full - Plot to see if dependence on invariant mass'''
c3, b3, p3 = plt.hist(bkg_full_cut, histtype = 'step', bins = bin(bkg_full_cut))
c4, b4, p4 = plt.hist(bkg, histtype = 'step', bins = bin(bkg))
plt.show()

err = np.sqrt(1/c4)* counts_bkg2/counts_bkg1
plt.errorbar(bins_bkg1[0:54], counts_bkg2/counts_bkg1, yerr = err, fmt='xr')
plt.xlabel('Invariant mass [MeV]', fontsize=25)
plt.annotate('ATLAS Work in Progress', xy = (143000, 1.12), fontsize=20)
'''Different positions for each network '''
#plt.annotate('ATLAS Work in Progress', xy = (145000, 1.027), fontsize=16)
#plt.annotate('ATLAS Work in Progress', xy = (145000, 1.28), fontsize=16)
#plt.annotate('ATLAS Work in Progress', xy = (145000, 5.4), fontsize=16)
plt.ylabel('Post cut background / total background  ', fontsize=25)
plt.title('Post cut counts / Full Background counts', fontsize=30)
plt.plot(b3[0:54], np.ones_like(b3[0:54]), linestyle = '--', color = 'b')
plt.xticks(fontsize = 20 )
plt.yticks(fontsize = 20 )
plt.show()

''' Scaled Histograms Signal and Background '''
counts1, bins1, p1 = plt.hist(sig, histtype = 'step', bins = bin(sig), weights = (2 * 8.1330E-5 * np.ones_like(sig)))
counts2, bins2, p2 = plt.hist(bkg, histtype = 'step', bins = bin(bkg), weights = (2 * 0.00189 * np.ones_like(bkg)))
plt.show()

print('Counts Sig, Bkg scaled')
print(bins1[16:25])
print(np.sum(counts1[16:25]))
print(bins2[16:25])
print(np.sum(counts2[16:25]))

''' Define exponential function for background'''
def exp(x, C,  a):
    exponential = C * np.exp(a * x)
    return  exponential 

''' Plot exponential and full scaled background and compare to get C, a '''
plt.plot( bins2[0:54]/1000 , counts2[0:55], color = 'b')
plt.plot( bins2[0:55]/1000 , exp(bins2[0:55]/1000, 50, -0.01), color = 'r')
plt.show()

''' Perform exponential fit to background '''
#minimises non-linear least squares 
params , covariance = curve_fit( exp , bins2[0:54]/1000 , counts2[0:55], p0=(50, -0.01) )
print('Parameters determined from curve_fit: ' + str(params))
#the diagonal value in the covariance array return the variance, therefore the standard deviation of each of these values is given by the square root of these values
print('stdev from curve_fit: ' + str(np.sqrt(np.diag(covariance)))) 

'''Plot exponential fit over background'''
plt.plot( bins2[0:55]/1000 , exp(bins2[0:55]/1000, params[0], params[1]), label = 'exponential fit')
plt.errorbar(bins2[0:54]/1000+ 0.5 * (bins2[1]/1000 - bins2[0]/1000), counts2[0:55], yerr = np.sqrt(2 * 0.00189 * counts2[0:54]), fmt='xr', ms = (5),  label = 'background invariant mass, probability > 0.7')
plt.xlabel('Invariant Mass [GeV]', fontsize=16)
plt.ylabel('Number of events', fontsize=16)
plt.title('Exponential fit to background invariant mass', fontsize=16)
plt.annotate('ATLAS Work in Progress', xy = (145, 9.64), fontsize = 16)
#plt.annotate('ATLAS Work in Progress', xy = (145, 7.48), fontsize = 16)
plt.xticks(fontsize = 12 )
plt.yticks(fontsize = 12 )
plt.legend(fontsize = 12)
plt.show()

''' Evaluate the integral of the exponential in the region 121-129 GeV = Background Counts '''
def bkg_counts():
    bkg_counts =  (1 / params[1]) * ( exp( bins2[25]/1000, params[0] , params[1] ) -  exp( bins2[16]/1000, params[0] , params[1] ) )
    return bkg_counts 

''' Plot signal, Background and Exponential fit in range 121-129 GeV to check '''
plt.plot( bins1[16:25]/1000 , counts1[16:25] )
plt.plot( bins2[16:25]/1000 , counts2[16:25] )
plt.plot( bins2[16:25]/1000 , exp(bins2[16:25]/1000, params[0], params[1]) )
plt.show()

'''Obtain the approximate sensitivity value'''
def sensitivity():

    S = np.sum(counts1[16:25])
    b1 = np.sum(counts2[16:25])
    B = bkg_counts()
    n = S + B

    #check reasonable values by printing to terminal
    print(b1)
    print(S)
    print(B)

    sensitivity =   np.sqrt(2 * (n * np.log(n/B) - S))

    return print('The sensitivity is: ' + str(sensitivity))

sensitivity()
