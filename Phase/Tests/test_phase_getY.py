#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:50:10 2022

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
     




if __name__=='__main__':
    '''
    Testing 
    '''
   
    
    
    
    
    ################################################################
    ################################################################
    seed = 1211
   
    np.random.seed(seed)
    
    delta = 0.1
    nspins = 6    
    num_layers =3
    num_itr =1
    machine_precision = 10  
    significant_figures = 2
    eta = 100
    shots = 10**(2*significant_figures)
    shots_amplitude = shots
    shots_phase = shots
    Q = getRandomQ(nspins)
    gamma= np.pi/10
    
    
    
    
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
    inputs['gamma']=gamma
    ####
    inputs['delta'] = delta
    
    
    print(pd.DataFrame([inputs], index=['input values']).T , '\n')
    
    circ_U = getRandomU(nspins, num_layers) 
    circ_Q = getQCirc(circ_U, Q)
    circ_UQU = getUQUCirc(circ_U, circ_Q)
    
    amplObj  = Amplitude(circ_U, circ_UQU, Q, shots_amplitude, eta, significant_figures)
    
    print(amplObj.df_count)
    print(amplObj.df_ampl)
    
    phaseObj = Phase( amplObj.df_ampl.copy(), nspins, amplObj.circ_U, amplObj.Q, 
                          significant_figures, shots_phase)
    
    
    ##############
    delta_k = delta
    
    phaseObj()
    
    [j_list, delta_times_cj_values] = phaseObj.getY(delta)
    
    # ys_j = phaseObj.yj_parameters
    # js = phaseObj.
    # cjs = phaseObj.cjs
    # # 
    
    
    
    
    