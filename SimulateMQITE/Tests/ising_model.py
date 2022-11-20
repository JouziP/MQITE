#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:54:48 2022

@author: pejmanjouzdani
"""

import pandas as pd
import numpy as np

from SimulateMQITE.Simulator import Simulator



if __name__=='__main__':
    
      
    
    seed = 12321    
    ########  seed random
    np.random.seed(seed)
    
    
    
    
    delta= 0.05
    T =10 * delta
    
    ############ one-D Ising model
    Qs = [
        [1,1, 0,0, 0,0],
        [0,1, 1,0, 0,0],        
        [0,0, 1,1, 0,0],        
        [0,0, 0,1, 1,0],        
        [0,0, 0,0, 1,1],        
        
        [3,0, 0,0, 0,0],
        [0,3, 0,0, 0,0],        
        [0,0, 3,0, 0,0],        
        [0,0, 0,3, 0,0],        
        [0,0, 0,0, 3,0],  
        [0,0, 0,0, 0,3],  
        
        
        ]
    
    
    
    ###########
    
    nspins = len(Qs[0])
    n_H = len(Qs)
    
    ws = [1 for q in range(nspins-1)] # J_z
    ws += [-1 for q in range(nspins)] # H_x
    
    ###########
    
    
    shots_amplitude = 10000000
    shots_phase = 10000
    eta =100
    significant_figures = 2
    machine_precision  = 10
    
         

        
    inputs={}
    inputs['seed']=seed
    inputs['nspins']=nspins        
    inputs['T']=T
    inputs['machine_precision']=machine_precision
    inputs['significant_figures']=significant_figures
    inputs['eta']=eta
    inputs['shots-amplitude']=shots_amplitude    
    inputs['shots-phase']=shots_phase    
    inputs['delta']=delta
    inputs['n_H']=n_H    
    
    
    print(pd.DataFrame([inputs], index=['input values']).T , '\n')
    
    
    
    #### Exact value
    from BasicFunctions.exact_calculations import getExact
    
    E_exact, _, H_mtx, ___ = getExact(Qs, ws)
    
    print('E exact = ' , E_exact)
    
    simulator = Simulator(T, n_H, delta, Qs, ws, nspins , shots_amplitude,
                  shots_phase ,  eta, significant_figures, 
                  machine_precision, 
                  observable_matrix=H_mtx,
                  observable_exact=E_exact)
    simulator()
    print(pd.DataFrame([inputs], index=['input values']).T , '\n')
    print('E exact = ' , E_exact)
            