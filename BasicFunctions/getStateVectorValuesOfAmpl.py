#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 17:10:42 2022

@author: pejmanjouzdani
"""



import numpy as np
import pandas as pd


from BasicFunctions.functions import getState




def getStateVectorValuesOfAmpl( j_list, circ_uhu, significant_figures, machine_precision):
        #################### Benchmark 
        circ_state = getState(circ_uhu, machine_precision)   
        
        ###        
        df_ampl_bm = pd.DataFrame(circ_state[j_list, :].round(significant_figures).T.tolist()[0])
        
        df_ampl_bm.columns=['|c_j|']
        df_ampl_bm.index = j_list
        
        
        #### not normalizing !
        df_ampl_bm['|c_j|'] = df_ampl_bm['|c_j|'].apply(lambda x: np.sqrt(x*x.conjugate()).real)

        return df_ampl_bm
    
    


if __name__=='__main__':
    
    from qiskit import QuantumCircuit

    from BasicFunctions.getQCirc import getQCirc
    from BasicFunctions.getRandomQ import getRandomQ
    from BasicFunctions.getRandomU import getRandomU
    from BasicFunctions.getUQUCirc import getUQUCirc
    
    seed = 1211
    np.random.seed(seed)
    
    ################################################################
    ################################################################
    nspins = 10    
    num_layers =4
    num_itr =1
    machine_precision = 10  
    significant_figures = 3 
    eta = 100
    shots = 10**(2*significant_figures)
    Q = getRandomQ(nspins)
    
    
    inputs={}
    inputs['nspins']=nspins
    inputs['num_layers']=num_layers
    inputs['num_itr']=num_itr
    inputs['machine_precision']=machine_precision
    inputs['significant_figures']=significant_figures
    inputs['eta']=eta
    inputs['shots']=shots    
    inputs['Q-list']=Q
    
    circ_U = getRandomU(nspins, num_layers)        
    circ_Q = getQCirc(circ_U, Q)
    circ_UQU = getUQUCirc(circ_U, circ_Q)
    
    print(pd.DataFrame([inputs]).T )
    
    
    from Amplitude.Amplitude import Amplitude
    
    df_count = Amplitude.getIndexsFromExecute(circ_UQU, shots)
    df_ampl  = Amplitude.getAmplitudes(df_count, eta)
    
    j_list = df_ampl.index.tolist()
    
    state_vec = getStateVectorValuesOfAmpl( j_list, circ_UQU, significant_figures, machine_precision)
    
    df_qc_vs_bm = pd.concat((df_ampl, state_vec), axis=1)
    df_qc_vs_bm.columns = ['QC', 'StVec']
    df_qc_vs_bm['diff'] = df_qc_vs_bm['QC'] - df_qc_vs_bm['StVec']    
    print(df_qc_vs_bm.round(significant_figures))
    print('sum diff^2 rounded to significant figures: ' ,
          np.round(  np.sqrt(sum(df_qc_vs_bm.round(significant_figures)['diff'].apply(lambda x: x**2))), significant_figures))
    
    