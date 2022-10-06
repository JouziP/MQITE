#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:33:48 2022

@author: pejmanjouzdani
"""





import numpy as np
import pandas as pd


from Amplitude.getIndexsFromExecute import getIndexsFromExecute
from Amplitude.getAmplitudes import getAmplitudes
from Amplitude.getBenchmark_before import getBenchmark_before
from Amplitude.getBenchmark_after import getBenchmark_after


class Amplitude:
    def __init__(self, circ, circ_uhu, shots, eta, significant_figures, machine_precision=10):
        self.circ = circ
        self.circ_uhu = circ_uhu
        self.shots = shots
        self.eta = eta
        self.significant_figures = significant_figures
        self.machine_precision = machine_precision
        
    def computeAmplutudes(self):
        
        #################### |cj|^2 and js upto significant digits - tested +
        [counts, j_indxs], df_count = self.getIndexsFromExecute(self.circ_uhu, self.shots)   
        
        ### some benchmarks before trimming by eta - tested +
        m_support_rounded, drop_in_peak, m_support, std_prob = self.getBenchmark_before( df_count, self.significant_figures)
        
        ### keep eta of them -> {|c_j|} = df_ampl - tested +
        j_list, df_ampl = self.getAmplitudes(df_count, self.eta)
        
        ### some other bm after trimming - tested 
        df_ampl_bm = self.getBenchmark_after( df_ampl, j_list, self.circ_uhu, self.significant_figures, self.machine_precision)
        print(df_ampl_bm)
        
        return df_ampl_bm, df_ampl, std_prob, drop_in_peak, m_support, m_support_rounded
        
        
        
    def getIndexsFromExecute(self, circ, shots, backend = 'qasm_simulator'):
        pass
        return  getIndexsFromExecute(circ, shots, backend)
        
    
    def getAmplitudes(self, df_count, eta):                
        pass
        return getAmplitudes( df_count, eta)
        
   
        
    def getBenchmark_before(self, df_count, significant_figures):
        return getBenchmark_before(df_count, significant_figures)
        
    def getBenchmark_after(self, df_ampl, j_list, circ, significant_figures, machine_precision):
        return getBenchmark_after( df_ampl, j_list, circ, significant_figures, machine_precision)
    
        
    


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
    
    ################################# TEST
    ###### Constructor
    myAmplObj = Amplitude(circ, circ_uhu, shots, eta, significant_figures, machine_precision=10)
    
    ###### main public method
    df_ampl_bm, df_ampl, std_prob, drop_in_peak, m_support, m_support_rounded =  myAmplObj.computeAmplutudes()
    df = pd.concat(
        (df_ampl.round(significant_figures), df_ampl_bm.round(significant_figures)), 
        axis = 1)
    
    df.columns = ['QC', 'SV']
    
    print()
    print('EXPLANATION OF THE TEST :')
    print('=========================')
    print('This test creates a random circuit with %s number of layers of gates, and %s number of qubits '%(num_layers, nspins))
    print('A Pauli string prator Q is randomly generated, X==0, Y==1, etc.  -->  Q = %s'%(Q))    
    print('The actual number of observed |c_j|^2 with non-zero values are %s '%(m_support))    
    print('The same number of observed |c_j|^2 with non-zero values AFTER rounding to %s significant figure is %s which is expected to be the same as above'%(significant_figures, m_support_rounded))    
    print('The resean is that the number of shots is supposed to genrate numbers that are significant upto the %s, so when rounding to it the non-zero values are the same before rounding '%significant_figures)    
    print()
    print('In practice we set a cut of eta: a maximum number of %s (eta) components with largest |c_j|^2 are retained'%(eta))    
    print('Amplitudes |c_j| from QC execution with %s shots, compared with the state vector (SV) values when rounded to %s significant figures'%(shots, significant_figures))    
    print(df)
    print('Notice the dimension of the table above is %s'%eta)    
    print('summing the square of the differences, we get the error')    
    error_avg = np.sum((df['QC'] - df['SV'] ).apply(lambda x: x**2)) 
    
    print('error_avg = ' , error_avg)
         
    