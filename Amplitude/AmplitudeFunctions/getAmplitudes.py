#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 17:04:53 2022

@author: pejmanjouzdani
"""



import numpy as np
import pandas as pd



def getAmplitudes( df_count, eta):                
        '''
        
         return the |c_j| of the most largest amplitudes
     
        '''
        if not isinstance(eta, int):
            raise TypeError(' eta is not an integer')
            
        
            
            
        ### amplitudes |cj|
        df_ampl = (df_count/df_count.sum()).apply(lambda x: np.sqrt(x)) ## - checked
        df_ampl = df_ampl.sort_values('n_j', ascending=False)
        
        ####################  pick the eta dominant amplitudes |cj|
        df_ampl = df_ampl[:eta]
        
        #####  making sure they are all non-zero
        df_ampl = df_ampl.loc[df_ampl['n_j']!=0]
        
        df_ampl.columns=['|c_j|']
        
        return df_ampl
    
    
    


if __name__=='__main__':
    pass
    from qiskit import QuantumCircuit

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
    
    seed = 1253
    np.random.seed(seed)
    
    ################################################################
    ################################################################
    nspins = 12    
    num_layers =2
    num_itr =1
    machine_precision = 10        
    shots = 1000
    eta = 100
    significant_figures = 3#np.log10(np.sqrt(shots)).astype(int)
    
    circ = getRandomU(nspins, num_layers)
    Q = getRandomQ(nspins)
    circ_uhu = getCircUQU(circ, Q)
    
    
    from Amplitude.getIndexsFromExecute import getIndexsFromExecute    
    [counts, j_indxs], df_count = getIndexsFromExecute(circ_uhu, shots, backend = 'qasm_simulator')    

    print(df_count)
    print()

    j_list, df_ampl = getAmplitudes( df_count, eta)
    print (df_ampl)
    