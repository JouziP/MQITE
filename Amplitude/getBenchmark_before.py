#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 17:08:54 2022

@author: pejmanjouzdani
"""




import numpy as np
import pandas as pd



def getBenchmark_before(df_count, significant_figures):
    pass

    ###     
    m_support = df_count.shape[0]    
    print('m_support = ', m_support)

    ### amplitudes |cj|
    df_ampl = (df_count/df_count.sum()).apply(lambda x: np.sqrt(x))  
    df_ampl = df_ampl.sort_values(0, ascending=False)
    df_ampl_org = pd.DataFrame.copy(df_ampl) # the amplitudes before rounding

    #################### some other metrics 
    m_support_rounded = pd.DataFrame.sum(df_ampl_org.round(significant_figures)!=0)[0]
    print('m_support_rounded = ', m_support_rounded)
    
    ###  the observable (metric) standard deviation 
    std_prob = df_ampl.std()[0]    
    if pd.isna(std_prob):
        std_prob = 0     
    print('std_prob = ', std_prob)
    
    ###  the observable (metric) \Delta^*
    try:
        drop_in_peak = df_ampl_org.values[0,0] - df_ampl_org.values[1,0]
    except:
        drop_in_peak = df_ampl_org.values[0,0]
        
    return m_support_rounded, drop_in_peak, m_support, std_prob





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
    
    
    ################################################################
    ################################################################
    from Amplitude.getIndexsFromExecute import getIndexsFromExecute
    
    [counts, j_indxs], df_count = getIndexsFromExecute(circ_uhu, shots, backend = 'qasm_simulator')
    
    m_support_rounded, drop_in_peak, m_support, std_prob = getBenchmark_before(df_count, significant_figures)