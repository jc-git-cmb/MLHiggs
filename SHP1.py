import numpy as np 
import math 
import pandas as pd 
import tensorflow as tf
from matplotlib import pyplot as plt 
from pylorentz import Momentum4
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from keras import backend as K
from sklearn.metrics import roc_curve
from keras.backend import clear_session


#import siganl data
f1 = 'ttH.csv'

#import main background data (tt-YY)
f2 = 'bkg_yy_tt.csv'


def Signal():
    # create counts for the number of columns in each event (row)
    with open(f1, mode = 'r') as sig:
        rows = sig.readlines()
        columns = [ len(i.split(",")) for i in rows ]

    #assign an index of 0 to max number of columns - 1, to each column 
    col_name = [ j for j in range(0, max(columns))]

    #number of events in signal file 
    events = sum( 1 for line in open(f1) )

    #use pandas to read in signal file 
    #signal = pd.read_csv( 'ttH_100k.csv', header=None, delimiter = "," , names = range(max(col_name)) )
    signal = pd.read_csv( 'ttH.csv', header=None, delimiter = "," , names = range(max(col_name)) )

    #assign each variable in the datafile 
    y1_pt = signal.iloc[:,0]
    y1_eta = signal.iloc[:,1]
    y1_phi = signal.iloc[:,2]
    y1_E = signal.iloc[:,3]

    y2_pt = signal.iloc[:,4]
    y2_eta = signal.iloc[:,5]
    y2_phi = signal.iloc[:,6]
    y2_E = signal.iloc[:,7]

    #determine the difference between phi's for both photons
    delta_phi_sig = []
    #number of jets for each event ( row in data file )
    N_sig = []

    for m in range( 0, events ):
        #delta_phi
        delta_phi_sig.append( signal.iloc[m,6] - signal.iloc[m,2] )
        #number of jets
        jets = (columns[m] - 8)/5
        N_sig.append(jets)
    
    #calculate the invariant mass of the signal photons 
    #create array of zeros - mass of photons is 0 
    m_1 = np.zeros(events)
    #define the 4-momentum vectors for each photon using pylorentz
    y1 = Momentum4.m_eta_phi_pt(  m_1 , y1_eta , y1_phi , y1_pt )
    y2 = Momentum4.m_eta_phi_pt( m_1 , y2_eta , y2_phi , y2_pt )

    #sum the 4-vectors of the two photons 
    parent = y1 + y2 
    #find the invariant mass of the two photons 
    m_yy = parent.m 

    return N_sig, signal, y1_pt, y1_eta, y1_phi, y1_E, y2_pt, y2_eta, y2_phi, y2_E, m_yy, delta_phi_sig, events

signal = Signal()[1]
Signal = Signal()

def bkg_main():
    # create counts for the number of columns in each event (row)
    with open(f2, mode = 'r') as bkg:
        rows_bkg = bkg.readlines()
        columns_bkg = [ len(m.split(",")) for m in rows_bkg ]

    #assign an index of 0 to max number of columns - 1 to each column 
    col_name_bkg = [ n for n in range(0, max(columns_bkg))]

    #use pandas to read in signal file 
    #background = pd.read_csv( 'bkg_yy_tt_100k.csv', header=None, delimiter = "," , names = range(max(col_name_bkg)) )
    background = pd.read_csv( 'bkg_yy_tt.csv', header=None, delimiter = "," , names = range(max(col_name_bkg)) )
    
    #assign each variable in the datafile 
    bkg1_pt = background.iloc[:,0]
    bkg1_eta = background.iloc[:,1]
    bkg1_phi = background.iloc[:,2]
    bkg1_E = background.iloc[:,3]

    bkg2_pt = background.iloc[:,4]
    bkg2_eta = background.iloc[:,5]
    bkg2_phi = background.iloc[:,6]
    bkg2_E = background.iloc[:,7]

    #number of events in main background file 
    bkg_events = sum( 1 for line in open(f2) )

    #determine the difference between phi's for both photons
    delta_phi_bkg = []
    #number of jets for each event ( row in data file )
    N_bkg = []

    for t in range( 0, bkg_events ):
        #delta_phi
        delta_phi_bkg.append( background.iloc[t,6] - background.iloc[t,2] )
        #number of jets
        jets_bkg = (columns_bkg[t] - 8)/5
        N_bkg.append(jets_bkg)

    #create array of zeros - mass of photons is 0  
    my_bkg = np.zeros( bkg_events )
    #define the 4-momentum vectors for each photon using pylorentz
    bkg1 = Momentum4.m_eta_phi_pt(  my_bkg , bkg1_eta , bkg1_phi , bkg1_pt )
    bkg2 = Momentum4.m_eta_phi_pt( my_bkg , bkg2_eta , bkg2_phi , bkg2_pt )

    #sum the 4-vectors of the two photons 
    parent2 = bkg1 + bkg2 
    #find the invariant mass of the two photons 
    inv_mass_bkg = parent2.m 

    return  N_bkg, background, bkg1_pt, bkg1_eta, bkg1_phi, bkg1_E, bkg2_pt, bkg2_eta, bkg2_phi, bkg2_E, inv_mass_bkg, delta_phi_bkg, bkg_events

background = bkg_main()[1]
bkg_main = bkg_main()


#Graphs for variables in the datafiles 

#get the weights for each bin 
def weights(m):
    sum_weights = float( len(m) )
    numerators = np.ones_like( m )
    return( numerators / sum_weights )

#create a bins function
def bin(data): 
    bin = (np.max(data)-np.min(data))/1000
    return np.int(bin)
print( bin( Signal[10] ) )

#plot invariant mass of the photons for the signal and main background 
def inv_mass():
    plt.hist( Signal[10]/1000, histtype = 'step' , bins = bin( Signal[10] ) , color = 'b' , label = 'sinal', weights = weights( Signal[10]) )
    plt.hist( bkg_main[10]/1000, histtype = 'step' , bins = bin( bkg_main[10] ) , color = 'r' , label = 'background' , weights = weights( bkg_main[10] ) )
    plt.xlim(0, 250)
    plt.xlabel('Invariant mass of photons, GeV')
    plt.ylabel(' Normalised events ')
    plt.annotate('ATLAS Work in Progress', xy = (180, 0.225))
    plt.legend()
    plt.title( ' Graph 1 : Signal and main Background - invariant mass ')
    
    return plt.show()

#plot the number of jets produced by the signal and main background 
def num_jets_sig():

    plt.hist( Signal[0] , histtype = 'step', bins = 1000,  color = 'b' , label = 'signal')
    #plt.hist( bkg_main[0], histtype = 'step' , bins = 1000, color = 'r' , label = 'background')
    plt.xlabel(' Number of jets ')
    plt.ylabel(' Events ')
    plt.annotate('ATLAS Work in Progress', xy = (14, 1e3))
    plt.legend()
    plt.title( ' Graph 2 : Number of Jets produced - Signal')
    
    return plt.show()

#plot the number of jets produced by the signal and main background 
def num_jets_bkg():

    #plt.hist( Signal[0] , histtype = 'step', bins = 1000,  color = 'b' , label = 'sinal')
    plt.hist( bkg_main[0], histtype = 'step' , bins = 1000, color = 'r' , label = 'background')
    plt.xlabel(' Number of jets ')
    plt.ylabel(' Events ')
    plt.annotate('ATLAS Work in Progress', xy = (26, 1e4))
    plt.legend()
    plt.title( ' Graph 3 : Number of Jets produced - Background')
    
    return plt.show()

#plot the number of jets produced by the signal and main background 
def log_num_jets():

    plt.hist( Signal[0] , histtype = 'step', bins = 1000,  color = 'b' , label = 'sinal')
    plt.hist( bkg_main[0], histtype = 'step' , bins = 1000, color = 'r' , label = 'background')
    plt.yscale("log")
    plt.xlabel(' Number of jets ')
    plt.ylabel(' Events ')
    plt.legend()
    plt.title( ' Graph 2.2 : Log Plot of Number of Jets produced ')
    
    return plt.show()

#plot eta for both photons for the signal and main background 
def eta_photons():

    plt.hist( Signal[3], histtype = 'step', bins = 100,  color = 'b' , label = 'sinal - photon 1' ) 
    plt.hist( bkg_main[3], histtype = 'step' , bins = 100, color = 'r' , label = 'background - photon 1' )
    plt.hist( Signal[7], histtype = 'step', bins = 100,  color = 'g' , label = 'sinal - photon 2' ) 
    plt.hist( bkg_main[7], histtype = 'step' , bins = 100, color = 'm' , label = 'background - photon 2' )
    plt.xlabel(' Eta ')
    plt.ylabel(' Events ')
    plt.legend()
    plt.title( ' Graph 3 : Eta of photons 1 and 2 in signal and main background ')
    
    return plt.show()

#plot delta phi between two photons for the signal and main background 
def delta_phi_photons():
    plt.hist( Signal[11], histtype = 'step', bins = 100,  color = 'b' , label = 'sinal' )
    plt.hist( bkg_main[11], histtype = 'step' , bins = 100, color = 'r' , label = 'background' )
    plt.xlabel(' Delta Phi ')
    plt.ylabel(' Events ')
    plt.legend()
    plt.title( ' Graph 4 : Delta Phi for the two photons in signal and main background ')

    return plt.show()

#plot phi for both photons for the signal and main background 
def phi_photons():
    plt.hist( Signal[4], histtype = 'step', bins = 100,  color = 'b' , label = 'sinal - photon 1' ) 
    plt.hist( bkg_main[4], histtype = 'step' , bins = 100, color = 'r' , label = 'background - photon 1' )
    plt.hist( Signal[8], histtype = 'step', bins = 100,  color = 'g' , label = 'sinal - photon 2' ) 
    plt.hist( bkg_main[8], histtype = 'step' , bins = 100, color = 'm' , label = 'background - photon 2' )
    plt.xlabel(' Phi ')
    plt.ylabel(' Events ')
    plt.legend()
    plt.title( ' Graph 5 : Phis of photons 1 and 2 in signal and main background ')

    return plt.show()

#plot energy for both photons for the signal and main background 
def E_photons():
    plt.hist( Signal[5], histtype = 'step', bins = 100,  color = 'b' , label = 'sinal - photon 1' ) 
    plt.hist( bkg_main[5], histtype = 'step' , bins = 100, color = 'r' , label = 'background - photon 1' )
    plt.hist( Signal[9], histtype = 'step', bins = 100,  color = 'g' , label = 'sinal - photon 2' ) 
    plt.hist( bkg_main[9], histtype = 'step' , bins = 100, color = 'm' , label = 'background - photon 2' )
    plt.xlabel(' Energy')
    plt.ylabel(' Events ')
    plt.legend()
    plt.title( ' Graph 6 : Energy of photons 1 and 2 in signal and main background ')

    return plt.show()

#plot transverse momentum for both photons for the signal  
def pt_photons():

    plt.hist( Signal[2], histtype = 'step', bins = 100,  color = 'b' , label = 'sinal - photon 1' ) 
    plt.hist( Signal[6], histtype = 'step', bins = 100,  color = 'g' , label = 'sinal - photon 2' ) 
    plt.xlabel(' Transverse Momentum [MeV] ')
    plt.ylabel(' Fraction of Events ')
    plt.legend()
    plt.title( ' Transverse momentum of photons 1 and 2 in signal ')
    plt.annotate('ATLAS Work in Progress', xy = (150000, 5000), fontsize=20)
    
    return plt.show()


#plot transverse momentum for jets for the signal and background - first 3 jets
def pt_3jets():
    plt.hist( signal.iloc[:, 8 ] , histtype = 'step', bins = 100,  color = 'b' , label = 'sinal - jet1' ) 
    plt.hist( signal.iloc[:, 13 ], histtype = 'step', bins = 100,  color = 'g' , label = 'sinal - jet2' )
    plt.hist( signal.iloc[:, 18 ], histtype = 'step', bins = 100,  color = 'c' , label = 'sinal - jet3' )
    plt.hist( background.iloc[:, 8], histtype = 'step' , bins = 100, color = 'r' , label = 'background - jet1' )
    plt.hist( background.iloc[:, 13], histtype = 'step' , bins = 100, color = 'm' , label = 'background -jet2' )
    plt.hist( background.iloc[:, 18], histtype = 'step' , bins = 100, color = 'y' , label = 'background - jet3' )
    plt.xlabel(' transverse momentum ')
    plt.ylabel(' Events ')
    plt.legend()
    plt.title( ' Graph 7 : Transverse momentum of jets 1-3 in signal and main background ')

    return plt.show()

#plot eta for jets for the signal and background - first 3 jets
def eta_3jets():
    plt.hist( signal.iloc[:, 9 ] , histtype = 'step', bins = 100,  color = 'b' , label = 'sinal - jet1' ) 
    plt.hist( signal.iloc[:, 14 ], histtype = 'step', bins = 100,  color = 'g' , label = 'sinal - jet2' )
    plt.hist( signal.iloc[:, 19 ], histtype = 'step', bins = 100,  color = 'c' , label = 'sinal - jet3' )
    plt.hist( background.iloc[:, 9], histtype = 'step' , bins = 100, color = 'r' , label = 'background - jet1' )
    plt.hist( background.iloc[:, 14], histtype = 'step' , bins = 100, color = 'm' , label = 'background -jet2' )
    plt.hist( background.iloc[:, 19], histtype = 'step' , bins = 100, color = 'y' , label = 'background - jet3' )
    plt.xlabel(' Eta ')
    plt.ylabel(' Events ')
    plt.legend()
    plt.title( ' Graph 8 : Eta of jets 1-3 in signal and main background ')

    return plt.show()

#plot phi for jets for the signal and background - first 3 jets
def phi_3jets():
    plt.hist( signal.iloc[:, 10 ] , histtype = 'step', bins = 100,  color = 'b' , label = 'sinal - jet1' ) 
    plt.hist( signal.iloc[:, 15 ], histtype = 'step', bins = 100,  color = 'g' , label = 'sinal - jet2' )
    plt.hist( signal.iloc[:, 20 ], histtype = 'step', bins = 100,  color = 'c' , label = 'sinal - jet3' )
    plt.hist( background.iloc[:, 10], histtype = 'step' , bins = 100, color = 'r' , label = 'background - jet1' )
    plt.hist( background.iloc[:, 15], histtype = 'step' , bins = 100, color = 'm' , label = 'background -jet2' )
    plt.hist( background.iloc[:, 20], histtype = 'step' , bins = 100, color = 'y' , label = 'background - jet3' )
    plt.xlabel(' Phi ')
    plt.ylabel(' Events ')
    plt.legend()
    plt.title( ' Graph 9 : Phi of jets 1-3 in signal and main background ')

    return plt.show()

#plot Energy for jets for the signal and background - first 3 jets
def E_3jets():
    plt.hist( signal.iloc[:, 11 ] , histtype = 'step', bins = 100,  color = 'b' , label = 'sinal - jet1' ) 
    plt.hist( signal.iloc[:, 16 ], histtype = 'step', bins = 100,  color = 'g' , label = 'sinal - jet2' )
    plt.hist( signal.iloc[:, 21 ], histtype = 'step', bins = 100,  color = 'c' , label = 'sinal - jet3' )
    plt.hist( background.iloc[:, 11], histtype = 'step' , bins = 100, color = 'r' , label = 'background - jet1' )
    plt.hist( background.iloc[:, 16], histtype = 'step' , bins = 100, color = 'm' , label = 'background -jet2' )
    plt.hist( background.iloc[:, 21], histtype = 'step' , bins = 100, color = 'y' , label = 'background - jet3' )
    plt.xlabel(' Energy ')
    plt.ylabel(' Events ')
    plt.legend()
    plt.title( ' Graph 10 : Energy of jets 1-3 in signal and main background ')

    return plt.show()

#plot DRL for jets for the signal and background - first 3 jets
def jet_3DRL():
    plt.hist( signal.iloc[:, 12 ] , histtype = 'step', bins = 100,  color = 'b' , label = 'sinal - jet1' ) 
    plt.hist( signal.iloc[:, 17 ], histtype = 'step', bins = 100,  color = 'g' , label = 'sinal - jet2' )
    plt.hist( signal.iloc[:, 22 ], histtype = 'step', bins = 100,  color = 'c' , label = 'sinal - jet3' )
    plt.hist( background.iloc[:, 12 ], histtype = 'step' , bins = 100, color = 'r' , label = 'background - jet1' )
    plt.hist( background.iloc[:, 17 ], histtype = 'step' , bins = 100, color = 'm' , label = 'background -jet2' )
    plt.hist( background.iloc[:, 22] , histtype = 'step' , bins = 100, color = 'y' , label = 'background - jet3' )
    plt.xlabel(' DRL ')
    plt.ylabel(' Events ')
    plt.legend()
    plt.title( ' Graph 11 : DRL of jets 1-3 in signal and main background ')

    return plt.show()

print('done')

#call on all the functions to obtain plots
inv_mass()
log_num_jets()
eta_photons()
delta_phi_photons()
phi_photons()
E_photons()
pt_3jets()
eta_3jets()
phi_3jets()
E_3jets()
jet_3DRL()



#ML - 1 - NN

#create a column of ones for signal events
sig_1 = np.ones( Signal[12] , order = 'F' ) 
d1 = pd.DataFrame(sig_1, columns = ['bin'])

#create a column of zeros for background events
bkg_0 = np.zeros( bkg_main[12] , order = 'F' )
d2 = pd.DataFrame( bkg_0, columns = ['bin'])

#add each column to signal and background dataframes - column 0 
sig_ML = pd.concat( [d1 , signal] , axis = 1 )
bkg_ML = pd.concat( [d2 , background] , axis = 1 ) 

#combine signal and background dataframes
data = pd.concat( [sig_ML, bkg_ML] )
#randimly shuffle rows
data_shuffle = data.sample( frac = 1 )
#fill NaN values with 999
dataset = data_shuffle.fillna(999)
'''This data set was saved to my desktop - 'dataset_new_nn1.csv''''

#new data set is shown below
dataset = pd.read_csv( 'dataset_new_nn1.csv', header = 0 , delimiter = "," )

