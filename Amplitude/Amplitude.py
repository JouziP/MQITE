#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:33:48 2022

@author: pejmanjouzdani
"""



from SimulateMQITE.log_config import logger

import numpy as np
import pandas as pd



from Amplitude.AmplitudeFunctions.getIndexsFromExecute import getIndexsFromExecute
from Amplitude.AmplitudeFunctions.getAmplitudes import getAmplitudes
from BasicFunctions.getStateVectorValuesOfAmpl import getStateVectorValuesOfAmpl


class Amplitude:
    def __init__(self, circ_Q, circ_UQU, shots_amplitude, eta, significant_figures, machine_precision=10):
        
        self.circ_Q = circ_Q
        self.circ_UQU = circ_UQU
        self.shots_amplitude = shots_amplitude
        self.eta = eta
        self.significant_figures = significant_figures
        self.machine_precision = machine_precision
        
    def __call__(self, ):
        
        
            df_ampl = self.computeAmplutudes()
            self.df_ampl = df_ampl
            
        # if TestLevel==1:
            
        #     logger.info('%s.%s '%(self.__class__.__name__, self.__call__.__name__) )
        #     logger.info('TestLevel %s: preparing state vector amplitudes'%(TestLevel) )
            
        #     ### from shots
        #     df_ampl = self.computeAmplutudes()
        #     self.df_ampl = df_ampl
            
        #     ### Delta metric
        #     df_ampl_sorted = df_ampl.sort_values(0, ascending=False)
        #     if df_ampl_sorted.shape[0]>1:
        #         self.Delta__high_2_low = df_ampl_sorted.values[0,0] - df_ampl_sorted.values[1,0]
        #         self.Delta__high_2_low_definition='The difference between the largest observed amlitude and the next largest amplitude'
        #         ### log it
        #         logger.debug('\n')
        #         logger.debug('Delta__high_2_low_definition = %s'%(self.Delta__high_2_low_definition) )
        #         logger.debug('Delta__high_2_low = %s'%(self.Delta__high_2_low) )
                
        #     ### bm 
        #     df_ampl_bm = getStateVectorValuesOfAmpl( df_ampl.index.tolist(),
        #                                          self.circ_UQU,
        #                                          self.significant_figures, 
        #                                          self.machine_precision)
        #     self.df_ampl_bm=df_ampl_bm
            
                        
        #     ### sort    
        #     df_ampl_bm = df_ampl_bm.sort_index()
        #     df_ampl =  df_ampl.sort_index()
                
        #     df_bm_vs_qc = pd.concat(
        #         (df_ampl.round(self.significant_figures), df_ampl_bm.round(self.significant_figures)), 
        #         axis = 1)
            
        #     df_bm_vs_qc.columns = ['QC', 'SV']
        #     self.df_bm_vs_qc = df_bm_vs_qc
            
        #     ### log it
        #     logger.debug('df_bm_vs_qc =\n\n %s \n\n'%(self.df_bm_vs_qc) )                        


        
        
        
        # self.df_ampl_bm = df_ampl_bm        
        # self.std_prob = std_prob
        # self.drop_in_peak = drop_in_peak
        # self.m_support = m_support
        # self.m_support_rounded = m_support_rounded
        
        
        
    def computeAmplutudes(self):
        
        
        #################### |cj|^2 and js upto significant digits 
        df_count = self.getIndexsFromExecute(self.circ_UQU, self.shots_amplitude)   
        
        
        ### keep eta of them -> {|c_j|} = df_ampl 
        ### the j's and c_j's from shots and observed bit string --> NO state vector
        df_ampl = self.getAmplitudes(df_count, self.eta)
        
        return df_ampl
    
    
        
        
    @staticmethod
    def getIndexsFromExecute( circ_UQU, shots, backend = 'qasm_simulator'):
        pass
        return  getIndexsFromExecute(circ_UQU, shots, backend)
        
    @staticmethod
    def getAmplitudes( df_count, eta): 
                 
        return getAmplitudes( df_count, eta)
        
   
    
        
    


if __name__=='__main__':
    pass

    from qiskit import QuantumCircuit
    from BasicFunctions.functions import   getQCirc, getUQUCirc
   
    
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
    
    seed = 1253
    np.random.seed(seed)
    
    ################################################################
    ################################################################
    nspins = 12    
    num_layers =2
    num_itr =1
    machine_precision = 10        
    shots_amplitude = 100000
    eta = 100
    significant_figures = 2#np.log10(np.sqrt(shots_amplitude)).astype(int)
    TestLevel =  0
    
    
    
    circ = getRandomU(nspins, num_layers)
    Q = getRandomQ(nspins)
    circ_Q = getQCirc(circ, Q)
    circ_UQU = getUQUCirc(circ, circ_Q)
    
    ################################# TEST
    ###### Constructor
    myAmplObj = Amplitude( circ_Q, circ_UQU, shots_amplitude, eta, significant_figures, machine_precision=10)
    
    ###### main public method
    myAmplObj(TestLevel=TestLevel)
    
    
    if TestLevel==1:
        df = myAmplObj.df_bm_vs_qc
        
        print()
        print('EXPLANATION OF THE TEST :')
        print('=========================')
        print('This test creates a random circuit with %s number of layers of gates, and %s number of qubits '%(num_layers, nspins))
        print('A Pauli string prator Q is randomly generated, X==0, Y==1, etc.  -->  Q = %s'%(Q))    
        
        
        print('The resean is that the number of shots is supposed to genrate numbers that are significant upto the %s, so when rounding to it the non-zero values are the same before rounding '%significant_figures)    
        print()
        print('In practice we set a cut of eta: a maximum number of %s (eta) components with largest |c_j|^2 are retained'%(eta))    
        print('Amplitudes |c_j| from QC execution with %s shots, compared with the state vector (SV) values when rounded to %s significant figures'%(shots_amplitude, significant_figures))    
        print(df)
        print('Notice the dimension of the table above is %s'%eta)    
        print('summing the square of the differences, we get the error')    
        error_avg = np.sum((df['QC'] - df['SV'] ).apply(lambda x: x**2)) 
        
        print('error_avg = ' , error_avg)
             
    