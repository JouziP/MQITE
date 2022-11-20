#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 18:12:10 2022

@author: pejmanjouzdani
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from BasicFunctions.getState import getState

class AmplitudeBenchmark:
    def __init__(self): 
        self.current_error =  0
        self.error_t = []
        
    def getAmplFromStateVector(self, amplObj, t, k ):
        
        self.df_count =  amplObj.df_count.copy()
        observed_js = self.df_count.index.tolist()
        self.current_observed_js = observed_js
        circ_UQU    = amplObj.circ_UQU
        
        stateVectorUQU = getState(circ_UQU, amplObj.machine_precision)
        
        ### actual components of circ_UQU
        all_df_ampl = [   np.sqrt((cj[0,0]*cj[0,0].conjugate()).real) for cj in stateVectorUQU]
        all_df_ampl = pd.DataFrame(all_df_ampl, columns=['all |c_j| in statvec'])                
        all_df_ampl = all_df_ampl.loc[all_df_ampl['all |c_j| in statvec']!=0]
        all_df_ampl = all_df_ampl.sort_values(by='all |c_j| in statvec')
        
        self.all_df_ampl = all_df_ampl
        
        
        ### cj from statevector
        cj_from_stateVector      = stateVectorUQU[observed_js, :]
        self.cj_from_stateVector = cj_from_stateVector
        #        
        df_ampl_from_stateVector = [   np.sqrt((cj[0,0]*cj[0,0].conjugate()).real) for cj in cj_from_stateVector]
        df_ampl_from_stateVector = pd.DataFrame(df_ampl_from_stateVector, columns=['|c_j| in statvec'])
        df_ampl_from_stateVector.index=observed_js
        df_ampl_from_stateVector = df_ampl_from_stateVector.sort_index()        
        
        self.df_ampl_from_stateVector = df_ampl_from_stateVector
        
        
        ### comparison
        self.df_ampl_vs_statevector = pd.concat( (self.df_ampl_from_stateVector, amplObj.df_ampl), axis = 1 )  
        self.df_ampl_vs_statevector = self.df_ampl_vs_statevector.sort_index()
        
        if self.df_ampl_vs_statevector['|c_j| in statvec'].isnull().values.any():
            # print([   np.sqrt((cj[0,0]*cj[0,0].conjugate()).real) for cj in cj_from_stateVector])
            # df_ampl_from_stateVector = [   np.sqrt((cj[0,0]*cj[0,0].conjugate()).real) for cj in cj_from_stateVector]
            # df_ampl_from_stateVector = pd.DataFrame(df_ampl_from_stateVector, columns=['|c_j| in statvec'])
            # df_ampl_from_stateVector.index=observed_js
            # print(df_ampl_from_stateVector)
            # df_ampl_from_stateVector = df_ampl_from_stateVector.sort_index()
            # print(df_ampl_from_stateVector)
            # df_ampl_from_stateVector = df_ampl_from_stateVector[:amplObj.eta]
            # print(df_ampl_from_stateVector)
            
            self.display_long()
            raise ValueError(' Something not right ')
        
        ### error
        error = self.df_ampl_vs_statevector['|c_j| in statvec'] - self.df_ampl_vs_statevector['|c_j|']
        
        self.current_error =  np.sqrt(error.apply(lambda x : x**2 ).sum()) / len(observed_js)
        
        self.error_t.append([t, k, self.current_error])
        
    def display(self, ):
        print('=============   From AmplitudeBenchmark ')
        print('------ df_count')
        print(self.df_count)
        print()        
        print('------ df_ampl_vs_statevector')
        print(self.df_ampl_vs_statevector)
        print()        
        print('------ df_ampl_from_stateVector')
        print(self.df_ampl_from_stateVector)
        print()        
        print('------ cj_from_stateVector')
        print(self.cj_from_stateVector)
        print()        
        print('------ all_df_ampl')
        print(self.all_df_ampl)
        print()        
        print('------ cumulative error /  number of js: ', self.current_error)
        print()
        print('------ observed js: ')
        print(np.array(self.current_observed_js) )
        print()
        print('------ all js: ')
        print(np.array(self.all_df_ampl.index.values) )
        print()
        
        fig, ax = plt.subplots(1,1 , figsize=[10, 8])
        error_t = pd.DataFrame(self.error_t)
        xs = [i for i in range(len(self.error_t))]
        ax.plot(xs, error_t[2].values)
        
        ax.set_xlabel('(t,k) step ')
        ax.set_ylabel('error  ')
        ax.legend()
        
        plt.show()
        plt.close()
    
    def display_long(self, ):
        print('=============   From AmplitudeBenchmark ')
        print('------ df_count')
        print(self.df_count)
        print()        
        print('------ df_ampl_vs_statevector')
        print(self.df_ampl_vs_statevector)
        print()        
        print('------ cumulative error /  number of js (due to normalization of the observed state): ', self.current_error)
        print()
        print('------ df_ampl_from_stateVector')
        print(self.df_ampl_from_stateVector)
        print()        
        print('------ cj_from_stateVector')
        print(self.cj_from_stateVector)
        print()        
        print('------ all_df_ampl')
        print(self.all_df_ampl)
        print()        
        
        print('------ observed js on QC processor by running UQU|0>: ')
        print(np.array(self.current_observed_js) )
        print()
        print('------ all js corresponding to non-zero entries in UQU|0> from state-vector: ')
        print(np.array(self.all_df_ampl.index.values) )
        print()
        
        
        
        
        
        




if __name__=='__main__':
    '''
    Testing 
    '''
   
    
    from BasicFunctions.getQCirc import getQCirc
    from BasicFunctions.getRandomQ import getRandomQ
    from BasicFunctions.getRandomU import getRandomU
    from BasicFunctions.getUQUCirc import getUQUCirc
    
       
    from Amplitude.amplitude import AmplitudeClass as Amplitude
    
    
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
    
    
    amplObjBenchmark  = AmplitudeBenchmark()
    
    
    amplObjBenchmark.getAmplFromStateVector(amplObj, t=0, k=0 )
    
    # print(amplObjBenchmark.df_ampl_vs_statevector)
    # print(amplObjBenchmark.current_error)
    
    amplObjBenchmark.display()