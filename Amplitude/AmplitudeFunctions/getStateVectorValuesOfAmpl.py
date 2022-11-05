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
        
        df_ampl_bm.columns=[0]
        df_ampl_bm.index = j_list
        
        df_ampl_bm[0] = df_ampl_bm[0].apply(lambda x: np.sqrt(x*x.conjugate()).real)

        return df_ampl_bm
    
    


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
    
    seed = 1211
    np.random.seed(seed)
    
    ################################################################
    ################################################################
    nspins = 10    
    num_layers =10
    num_itr =1
    machine_precision = 10  
    significant_figures = 3 
    eta = 100
    shots = 10**(2*significant_figures)

    circ = getRandomU(nspins, num_layers)
    Q = getRandomQ(nspins)
    circ_uhu = getCircUQU(circ, Q)
    
    
    from Amplitude.AmplitudeFunctions.getIndexsFromExecute import getIndexsFromExecute    
    from Amplitude.AmplitudeFunctions.getAmplitudes import getAmplitudes    
    
    df_count = getIndexsFromExecute(circ_uhu, shots, backend = 'qasm_simulator')       
    
    df_ampl = getAmplitudes(df_count, eta)
    
    
    df_ampl_bm = getStateVectorValuesOfAmpl( df_ampl.index.tolist(),
                                                 circ_uhu,
                                                 significant_figures, 
                                                 machine_precision)



    
    ### sort    
    df_ampl_bm = df_ampl_bm.sort_index()
    df_ampl =  df_ampl.sort_index()
        
    df = pd.concat(
        (df_ampl.round(significant_figures), df_ampl_bm.round(significant_figures)), 
        axis = 1)
    
    df.columns = ['QC', 'SV']
    print()
    print('EXPLANATION OF THE TEST :')
    print('=========================')
    print('This test creates a random circuit with %s number of layers of gates, and %s number of qubits '%(num_layers, nspins))
    print('A Pauli string prator Q is randomly generated, X==0, Y==1, etc.  -->  Q = %s'%(Q))    
    
    print('The resean is that the number of shots is supposed to genrate numbers that are significant upto the %s, so when rounding to it the non-zero values are the same before rounding '%significant_figures)    
    print()
    print('In practice we set a cut of eta: a maximum number of %s (eta) components with largest |c_j|^2 are retained'%(eta))    
    print('Amplitudes |c_j| from QC execution with %s shots, compared with the state vector (SV) values when rounded to %s significant figures'%(shots, significant_figures))    
    print(df)
    print('Notice the dimension of the table above is %s'%eta)    
    print('summing the square of the differences, we get the error')    
    error_avg = np.sum((df['QC'] - df['SV'] ).apply(lambda x: x**2)) 
    
    print('error_avg = ' , error_avg)
         


    