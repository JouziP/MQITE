#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:39:39 2022

@author: pejmanjouzdani
"""


import numpy as np
import pandas as pd


from qiskit import QuantumCircuit,  transpile
from qiskit import Aer

# from Amplitude.AmplitudeFunctions.getAmplitudes import getAmplitudes
# from Amplitude.AmplitudeFunctions.getIndexsFromExecute import getIndexsFromExecute

from Amplitude.amplitude import AmplitudeClass as Amplitude

def computeAmplFromShots(circ, shots, j_ref, backend = 'qasm_simulator'):
    pass
    '''
    
    
    '''
   #### FIRST ATTEMPT
    df_count = Amplitude.getIndexsFromExecute(circ, shots, backend)    
    try:
        
        m1 = df_count['n_j'][j_ref]/shots #df_count[0].sum()           
        # print('------------ RESOLVED AT {shots} SHOTS; FIRST ATTEMPT ---------')
    except KeyError :
        print('WARNING: FIRST CIRCUIT RUN FAILED !')
        print('j_ref is not oobserved in the bitstrings... running the circuit with increased shots')
        print(f"Increasing shots from {shots} to  {10 * shots}")      
        print()
        
        #### SECOND ATTEMPT -> INCREASE THE SHOTS
        df_count = Amplitude.getIndexsFromExecute(circ, 10*shots, backend)
        try:
            m1 = df_count['n_j'][j_ref]/shots #df_count[0].sum()
            print('------------ RESOLVED AT {10*shots} SHOTS; SECOND ATTEMPT ---------')
        except KeyError:
            print('WARNING: SECOND CIRCUIT RUN FAILED !')
            print('j_ref is not oobserved in the bitstrings in the SECOND circuit run...')
            print(f"Increasing shots from  {10 * shots} to  {100*shots}")        
            print()
            
            #### THIRD ATTEMPT --> INCREASE SHOTS
            df_count = Amplitude.getIndexsFromExecute(circ, 100*shots, backend)
            try:
                m1 = df_count['n_j'][j_ref]/shots #df_count[0].sum()
                print('------------ RESOLVED AT {100*shots} SHOTS; THIRD ATTEMPT ---------')
            except:
                print('WARNING: THIRD CIRCUIT RUN FAILED !')
                print('j_ref is not oobserved in the bitstrings in the THIRD circuit run...')
                print(f"Increasing shots from {100 *  shots} to  {1000 * shots}")        
                print()
                
                #### FORTH ATTEMPT --> INCREASE SHOTS
                df_count = Amplitude.getIndexsFromExecute(circ, 1000*shots, backend) 
                try:
                    m1 = df_count['n_j'][j_ref]/shots #df_count[0].sum()
                    print('------------ RESOLVED AT {1000*shots} SHOTS; FORTH ATTEMPT ---------')
                except KeyError:
                    raise
     
                
            
    
    return m1


if __name__=='__main__':
    '''
    Testing the function:
        create a circ_UQU circuit 
        
    
    '''
    
    from qiskit import QuantumCircuit
    
    from Amplitude.amplitude import AmplitudeClass as Amplitude
    from BasicFunctions.functions import getBinary, getState
    from MultiQubitGate.functions import multiqubit
    
    ################################################################
    ################################################################
    ##################### FOR TEST           #######################
    ################################################################
    ################################################################
    ################################################################

    ################################################################
    ################################################################
    def getRandomU(nspins, num_layers=10):
        circ = QuantumCircuit(nspins)
        for l in range(num_layers):
            for i in range(nspins):
                ##############
                q=np.random.randint(nspins)
                g=np.random.randint(1, 4)
                p=np.random.uniform(-1,1)
                if g==1:
                    circ.rx(p,q)
                if g==2:
                    circ.ry(p,q)
                if g==2:
                    circ.rz(p,q)
                ##############
                q=np.random.randint(nspins-1)
                circ.cnot(q, q+1)
        return circ            
    
    ################################################################
    ################################################################
    def getRandomQ(nspins):
        Q = np.random.randint(0,2, size=nspins)
        return Q
    
    ################################################################
    ################################################################
    def getCircUQU(circ, Q):
        circ_uhu=QuantumCircuit.copy(circ)
        for (q,o) in enumerate(Q):
            if o==0:
                pass
            if o==1:
                circ_uhu.x(q)
            if o==2:
                circ_uhu.y(q)
            if o==3:
                circ_uhu.z(q)     
                
        
        circ_uhu=circ_uhu.compose(QuantumCircuit.copy(circ).inverse())
        return circ_uhu
    
    ################################################################
    ################################################################
    
    seed = 1211
    np.random.seed(seed)
    
    ################################################################
    ################################################################
    nspins = 2    # >=2
    num_layers =2
    num_itr =1
    machine_precision = 10        
    shots = 1000000
    eta = 100
    significant_figures = 3#np.log10(np.sqrt(shots)).astype(int)
    
    circ_U = getRandomU(nspins, num_layers)    
    Q = getRandomQ(nspins)
    circ_UQU = getCircUQU(circ_U, Q)
    
    print(Q)
    
    