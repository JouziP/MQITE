#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 09:57:37 2022

@author: pejmanjouzdani
"""



import pandas as pd
import numpy as np

from BasicFunctions.getQCirc import getQCirc
from BasicFunctions.getRandomQ import getRandomQ
from BasicFunctions.getRandomU import getRandomU
from BasicFunctions.getUQUCirc import getUQUCirc
from BasicFunctions.getStateVectorValuesOfAmpl import getStateVectorValuesOfAmpl



from Amplitude.Amplitude import Amplitude



class TestPhase:
    def __init__(self, **inputs):
        
        print(pd.DataFrame([inputs]).T , '\n')
        
        #####
        self.seed_for_random = inputs['seed']
        np.random.seed(self.seed_for_random)
        #####
        self.nspins = inputs['nspins']    
        self.num_layers =inputs['num_layers']
        self.num_itr =inputs['num_itr']
        self.machine_precision = inputs['machine_precision']  
        self.significant_figures = inputs['significant_figures'] 
        self.eta = inputs['eta']
        self.shots = inputs['shots']
        self.Q = inputs['Q']
        
     
        
    def test1(self, ):
        '''
        
        
        there may be cases where the number of obseved bit strings are more
        than eta !
        '''
        
        print('TEST 1 : GENERAL SANITY CHECK ')
        
        self.circ_U = getRandomU(nspins, num_layers)        
        self.circ_Q = getQCirc(self.circ_U, self.Q)
        self.circ_UQU = getUQUCirc(self.circ_U, self.circ_Q)
          
    def test2(self, ):
        '''
        
        '''
        
        print('\n\nTEST 2: GENERAL UNIT TEST of METHODS ')
        
        
        
if __name__=='__main__':
    '''
    Testing 
    '''
   
    
    
    
    
    ################################################################
    ################################################################
    seed = 1211
    nspins = 9    
    num_layers =3
    num_itr =1
    machine_precision = 10  
    significant_figures = 3 
    eta = 100
    shots = 10**(2*significant_figures)
    Q = getRandomQ(nspins)
    
    
    inputs={}
    inputs['seed']=seed
    inputs['nspins']=nspins
    inputs['num_layers']=num_layers
    inputs['num_itr']=num_itr
    inputs['machine_precision']=machine_precision
    inputs['significant_figures']=significant_figures
    inputs['eta']=eta
    inputs['shots']=shots    
    inputs['Q']=Q
    