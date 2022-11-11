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
from BasicFunctions.getState import getState




from Amplitude.amplitude import AmplitudeClass as Amplitude
from Phase.phase import Phase


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
        self.shots_amplitude = inputs['shots-amplitude']
        self.shots_phase = inputs['shots-phase']
        self.Q = inputs['Q']
        
    
        
        
    def test2(self, amplObj, circ_UQU, circ_U, Q):        
        '''
        
        '''
        
        self.df_ampl = pd.DataFrame.copy(amplObj.df_ampl)
        js = df_ampl.index.tolist()
        state_vec= getState(circ_UQU, self.machine_precision) 
        state_vec_js = state_vec[js, :]
        self.df_ampl_benchmark = pd.DataFrame([ np.sqrt(x[0,0]  *x[0,0].conjugate()).real for x in state_vec_js ])
        self.df_ampl_benchmark.columns=['n_j']
        self.df_ampl_benchmark.index=js
        self.df = pd.concat((self.df_ampl, self.df_ampl_benchmark), axis=1)
        self.df.columns=['qc','stvec']
        
        self.phaseObj = Phase(amplObj, 
                              self.nspins, 
                              circ_U, Q, self.significant_figures, self.shots_phase, )
        
        
if __name__=='__main__':
    '''
    Testing 
    '''
   
    
    
    
    
    ################################################################
    ################################################################
    seed = 1211
    nspins = 9    
    num_layers =2
    num_itr =1
    machine_precision = 10  
    significant_figures = 3 
    eta = 100
    shots = 10**(2*significant_figures)
    shots_amplitude = shots
    shots_phase = shots
    Q = getRandomQ(nspins)
    
    
    
    
    
    inputs={}
    inputs['seed']=seed
    inputs['nspins']=nspins
    inputs['num_layers']=num_layers
    inputs['num_itr']=num_itr
    inputs['machine_precision']=machine_precision
    inputs['significant_figures']=significant_figures
    inputs['eta']=eta
    inputs['shots-amplitude']=shots_amplitude    
    inputs['shots-phase']=shots_phase    
    inputs['Q']=Q
    
    testPhase = TestPhase(**inputs)
    circ_U = getRandomU(nspins, num_layers) 
    circ_Q = getQCirc(circ_U, Q)
    circ_UQU = getUQUCirc(circ_U, circ_Q)
    
    amplObj  = Amplitude(circ_U, circ_UQU, shots_amplitude, eta, significant_figures)
    
    df_ampl = amplObj.df_ampl
    print(df_ampl)
    testPhase.test2(amplObj, circ_UQU,  circ_U, Q)
    print(testPhase.df)
    
    
    
    