import seaborn as sns
import matplotlib 
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd
from pylorentz import Momentum4


#import siganl data
f1 = 'ttH.csv'

#import main background data (tt-YY)
f2 = 'bkg_yy_tt.csv'

''' Define column names '''
def column_names(file):

    col_yy1 = ['m_yy', 'y1_pt', 'y1_eta' , 'y1_phi' , 'y1_E' , 'y2_pt' , 'y2_eta' , 'y2_phi' , 'y2_E' ]
    col_yy = np.asarray(col_yy1)
    
    # create counts for the number of columns in each event (row)
    with open(file, mode = 'r') as file1:
        rows = file1.readlines()
        columns = [ len(i.split(",")) for i in rows ]

    col_jets1 = []

    #assign column names 
    for n in range( 1, max(columns)-8 ):
        col_jet_pt = "Jet{}_pt".format(n)
        col_jet_eta = "Jet{}_eta".format(n)
        col_jet_phi = "Jet{}_phi".format(n)
        col_jet_E = "Jet{}_E".format(n)
        col_jet_DRL = "Jet{}_DRL".format(n)
        col_jets1.append( col_jet_pt )
        col_jets1.append( col_jet_eta )
        col_jets1.append( col_jet_phi )
        col_jets1.append( col_jet_E)
        col_jets1.append( col_jet_DRL )
    
    col_jets = np.asarray(col_jets1)
    
    return np.concatenate( (col_yy, col_jets) , axis = 0 )

''' Find number of columns '''
def num_columns(file):
    # create counts for the number of columns in each event (row)
    with open(file, mode = 'r') as dat:
        rows = dat.readlines()
        columns = [ len(i.split(",")) for i in rows ]

    #assign an index of 0 to max number of columns - 1, to each column 
    col_name = [ j for j in range(0, max(columns))]
    return col_name

col_sig = num_columns(f1)
col_bkg = num_columns(f2)
print('done')

#read in data files using pandas
signal = pd.read_csv( 'ttH.csv', header=None, delimiter = ",", names = col_sig)
background = pd.read_csv( 'bkg_yy_tt.csv', header=None, delimiter = ",", names = col_bkg)

'''find invariant mass for background events'''
def bkg_mass():

    #assign each variable in the datafile 
    bkg1_pt = background.iloc[:,0]
    bkg1_eta = background.iloc[:,1]
    bkg1_phi = background.iloc[:,2]
    bkg1_E = background.iloc[:,3]

    bkg2_pt = background.iloc[:,4]
    bkg2_eta = background.iloc[:,5]
    bkg2_phi = background.iloc[:,6]
    bkg2_E = background.iloc[:,7]

    #create array of zeros - mass of photons is 0  
    my_bkg = np.zeros( (len(background)) )
    #define the 4-momentum vectors for each photon using pylorentz
    bkg1 = Momentum4.m_eta_phi_pt(  my_bkg , bkg1_eta , bkg1_phi , bkg1_pt )
    bkg2 = Momentum4.m_eta_phi_pt( my_bkg , bkg2_eta , bkg2_phi , bkg2_pt )

    #sum the 4-vectors of the two photons 
    parent2 = bkg1 + bkg2 
    #find the invariant mass of the two photons 
    inv_mass_bkg = parent2.m 

    return (pd.DataFrame(inv_mass_bkg))

inv_mass_dframe = bkg_mass()
print(inv_mass_dframe)

'''Define the two data frames to use - 1 for signal and background & one for background and invariant masses '''
combined = pd.concat( [signal.iloc[0:100000, 0:8] , background.iloc[0:1000000, 0:8]] , axis = 1 )
combined2 = pd.concat( [inv_mass_dframe.iloc[0:1000000] , background.iloc[0:1000000, 0:8]] , axis = 1 )
print(combined2)
print(combined)

''' Correlation'''
def correlation( dataset, file ):

    #default uses pearson method
    #finding correlation for background photons and invariant mass - easily altered for other options
    corr = dataset.iloc[0:500000, 0:8].corr()
    names = column_names(file)
    name1 = names[0:9]

    #use for combined signal and background
    comb_names = np.concatenate((name1, name2), axis = 0)

    #return a heatmap of the correlation using seaborn 
    return sns.heatmap(corr, xticklabels = name1, yticklabels = name1, cmap = 'coolwarm', annot=True, fmt= '.1f')
    
''' Call on for background files '''
correlation( combined2 , f1 )
plt.title('ATLAS Work in Progress: Correlations of Background Photons')
plt.show()