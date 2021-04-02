import numpy as np 
import math 
import pandas as pd 
from matplotlib import pyplot as plt 
from pylorentz import Momentum4

''' Each file was saved to my desktop '''

f1 = 'ttH.csv'
f2 = 'bkg_yy_tt_sameassig.csv'

dataset_orig = pd.read_csv( 'dataset_new_nn1.csv', header = 0 , delimiter = "," )


#input dataset and determine 1/2 length and 3/4 length (of number of rows/events)
def cols(file):

    with open(file, mode = 'r') as f:
        rows = f.readlines()
        columns = [ len(i.split(",")) for i in rows ]

    return max(columns)

''' Even signal and Backgrund Data sets '''
#make a dataset for the ANN with even number of signal and bkg events 
def short_data():

    signal = pd.read_csv( 'ttH.csv', header=None, delimiter = "," , names = range(cols(f1)) )
    background = pd.read_csv( 'bkg_yy_tt_sameassig.csv', header=None, delimiter = "," , names = range(cols(f2)) )

    ones = np.ones( len( signal ) , order = 'F' ) 
    d1 = pd.DataFrame( ones , columns = ['bin'])

    zero = np.zeros( len( signal ) , order = 'F' )
    d2 = pd.DataFrame( zero , columns = ['bin'])

    sig_ML = pd.concat( [d1 , signal] , axis = 1 )
    bkg_ML = pd.concat( [d2 , background] , axis = 1 ) 

    #join the datasets
    data = pd.concat( [sig_ML, bkg_ML] )
    #randomise lines in datasets
    data_shuffle = data.sample( frac = 1 )

    #fill nan values 
    dataset = data_shuffle.fillna(999)

    return dataset

dataset = short_data()
print('done')

def invmass():
    #define variables for 4 Momenta calculation 
    y1_eta = dataset.iloc[ :, 2 ]
    y1_phi = dataset.iloc[ :, 3]
    y1_pt = dataset.iloc[ :, 1 ]

    y2_eta2 = dataset.iloc[ :, 6 ]
    y2_phi2 = dataset.iloc[ :, 7]
    y2_pt2 = dataset.iloc[ :, 5 ]

    #create array of zeros - mass of photons is 0  
    zeros = np.zeros_like( y1_eta )
    #define the 4-momentum vectors for each photon using pylorentz
    y1 = Momentum4.m_eta_phi_pt(  zeros , y1_eta , y1_phi , y1_pt )
    y2 = Momentum4.m_eta_phi_pt( zeros , y2_eta2 , y2_phi2 , y2_pt2 )

    #sum the 4-vectors of the two photons 
    parent_mom = y1 + y2 

    #find the invariant mass of the two photons 
    inv_mass_yy = parent_mom.m 

    return inv_mass_yy

inv_mass_dframe = pd.DataFrame( invmass() )
print( inv_mass_dframe )
print('done')

''' Determine invariant mass values for full data set '''
def invmass_full():

    #define variables for 4 Momenta calculation 
    y1_eta_o = dataset_orig.iloc[ :, 2 ]
    y1_phi_o = dataset_orig.iloc[ :, 3]
    y1_pt_o = dataset_orig.iloc[ :, 1 ]

    y2_eta2_o = dataset_orig.iloc[ :, 6 ]
    y2_phi2_o = dataset_orig.iloc[ :, 7]
    y2_pt2_o = dataset_orig.iloc[ :, 5 ]

    #create array of zeros - mass of photons is 0  
    zeros = np.zeros_like( y1_eta_o )

    #define the 4-momentum vectors for each photon using pylorentz
    y1_o = Momentum4.m_eta_phi_pt(  zeros , y1_eta_o , y1_phi_o , y1_pt_o )
    y2_o = Momentum4.m_eta_phi_pt( zeros , y2_eta2_o , y2_phi2_o , y2_pt2_o )

    #sum the 4-vectors of the two photons 
    parent_mom_o = y1_o + y2_o 

    #find the invariant mass of the two photons 
    inv_mass_yy_o = parent_mom_o.m 

    return inv_mass_yy_o

inv_mass_dframe_full = pd.DataFrame( invmass_full() )
print('done')

''' Determine comparison model data'''
#define a function to divide pt and E by m_yy in the datafile and save it so it can be used for comparison with the ANN
def comparison_data():

    dat1 = dataset_orig.iloc[:, 1] / inv_mass_full.iloc[:, 0]
    dat2 = dataset_orig.iloc[:, 4] / inv_mass_dframe.iloc[:, 0]
    dat3 = dataset_orig.iloc[:, 5] / inv_mass_dframe.iloc[:, 0]
    dat4 = dataset_orig.iloc[:, 8] / inv_mass_dframe.iloc[:, 0]
    
    comparison_data = pd.concat( [dataset_orig.iloc[:,0] , dat1 , dataset_orig.iloc[ :, 2:3], dat2 , dat3, dataset_orig.iloc[:, 6:7], dat4, dataset_orig.iloc[:,-1]] )

    return comparison_data

#checks
new = comparison_data()
print(new.iloc[0:10, 0:8])
print(dataset_orig.iloc[0:10, 0:8])
print(inv_mass_dframe_full[0:10])
print('done')

