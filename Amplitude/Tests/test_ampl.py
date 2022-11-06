#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 12:28:38 2022

@author: pejmanjouzdani
"""

import pandas as pd
import numpy as np

from BasicFunctions.getQCirc import getQCirc
from BasicFunctions.getRandomQ import getRandomQ
from BasicFunctions.getRandomU import getRandomU
from BasicFunctions.getUQUCirc import getUQUCirc
from BasicFunctions.getStateVectorValuesOfAmpl import getStateVectorValuesOfAmpl



from Amplitude.Amplitude import Amplitude


class TestAmplitude:
    def __init__(self, **inputs):
        
        print(pd.DataFrame([inputs]).T , '\n')
        
        #####
        self.seed_for_random = inputs['seed']
        np.random.seed(self.seed_for_random)
        #####
        self.nspins = inputs['nspins']    
        self.num_layers =inputs['num_layers']
        self.num_itr =inputs['num_itr']
        self.machine_precision = inputs['machine_precision']  
        self.significant_figures = inputs['significant_figures'] 
        self.eta = inputs['eta']
        self.shots = inputs['shots']
        self.Q = inputs['Q']
        
        
        
        
    def test1(self, ):
        '''
        Tests he function Amplitude.getIndexsFromExecute
        
        there may be cases where the number of obseved bit strings are more
        than eta !
        '''
        
        print('TEST 1 : GENERAL SANITY CHECK ')
        
        self.circ_U = getRandomU(nspins, num_layers)        
        self.circ_Q = getQCirc(self.circ_U, self.Q)
        self.circ_UQU = getUQUCirc(self.circ_U, self.circ_Q)
        
        
        df_count      = Amplitude.getIndexsFromExecute(self.circ_UQU, self.shots)
        self.df_count = pd.DataFrame.copy(df_count)
        ############
        
        ####   Assertion:  if not True test 1 returns False
        print('\n\nPART 1 :  ASSERTION ')
        if df_count['n_j'].sum() != self.shots:            
            print(  'df_count["n_j"].shape[0] = ',  df_count['n_j'].shape[0])
            print('eta ', self.eta)            
            print('shots ',self.shots)
            print('\n')
            return False
        else:
            print('PART 1 PASSED\n')
        
        ####
        print('PART 2 : WARNING NOT ASSERTION')
        if df_count['n_j'].shape[0] > self.eta:
            print('Warning: number of {j} observed is more than threshold eta ! ')
    
            print(  'df_count["n_j"].shape[0] = ',  df_count['n_j'].shape[0])
            print('eta ', self.eta)            
            print('shots ',self.shots)
            print('\n')
        else:
            print('PART 2 PASSED\n')
            
        
        return True
        
        
        
    def test2(self, ):
        '''
        Tests he function Amplitude.
        '''
        
        print('\n\nTEST 2: GENERAL UNIT TEST AMPLITUDE METHODS ')
        
        if self.test1():
            try:
                df_ampl = Amplitude.getAmplitudes(self.df_count, self.eta)
            except:
                print('Something not right with Amplitude.getAmplitudes')
                return False
            
            self.df_ampl = pd.DataFrame.copy(df_ampl)
            
            ### check if the \cum_j |c_j|^2 = 1
            norm = sum(df_ampl['|c_j|'].apply(lambda x: x**2))
            norm = np.round(norm, self.significant_figures)
            
            if norm !=1:
                print('WARNING !')
                print('\cum_j |c_j|^2 \\neq 1 \n')
                print('\cum_j |c_j|^2 = ', norm, '\n')
            
            ### get the values from statevector and compare
            j_list = df_ampl.index.tolist()
            self.state_vector_c_j_for_js = getStateVectorValuesOfAmpl(j_list, 
                                       self.circ_UQU,
                                       self.significant_figures,
                                       self.machine_precision)
            
            self.df_ampl_vs_StVec = pd.concat((df_ampl, self.state_vector_c_j_for_js), axis =1 )
            self.df_ampl_vs_StVec.columns=['QC_|c_j|', 'StVec_|c_j|']
            self.df_ampl_vs_StVec['diff'] = self.df_ampl_vs_StVec['QC_|c_j|'] - self.df_ampl_vs_StVec['StVec_|c_j|']    
            
            
            print(self.df_ampl_vs_StVec.round(significant_figures), '\n\n')
            
        
    def test3(self, ):
        '''
        Tests Final target Amplitude.
        '''
        print('\n\nTEST 3 : AMPLITUDE FROM QC. VS STATE_VECTOR AMPLITUDES ')
        
        self.circ_U = getRandomU(nspins, num_layers)        
        self.circ_Q = getQCirc(self.circ_U, self.Q)
        self.circ_UQU = getUQUCirc(self.circ_U, self.circ_Q)
        
        
        amplObj = Amplitude(self.circ_Q, self.circ_UQU, self.shots, self.eta, self.significant_figures)
        
        amplObj()
        
        ### The |c_j| observed from execution of the ciruicts
        df_ampl = amplObj.df_ampl
        
        ### check if they match the statevector ampl
        j_list = df_ampl.index.tolist()        
        state_vector_c_j_for_js = getStateVectorValuesOfAmpl(j_list, self.circ_UQU, self.significant_figures, self.machine_precision)
        
        df_ampl_vs_StVec = pd.concat((df_ampl, state_vector_c_j_for_js), axis =1 )
        df_ampl_vs_StVec.columns=['QC_|c_j|', 'StVec_|c_j|']
        df_ampl_vs_StVec['diff'] = df_ampl_vs_StVec['QC_|c_j|'] - df_ampl_vs_StVec['StVec_|c_j|']
        self.df_ampl_vs_StVec = df_ampl_vs_StVec
        
        error = np.round(  np.sqrt(sum(df_ampl_vs_StVec.round(self.significant_figures)['diff'].apply(lambda x: x**2))), self.significant_figures)
        print('error = ', error , '\n')
        
        print('df_ampl_vs_StVec \n')
        print(df_ampl_vs_StVec.round(self.significant_figures), '\n')
        
        
        
        error_theshold = 10 ** (-1* (self.significant_figures-1))
        print('error_hreshold (= 10^-(\epsilon-1) ) : ', error_theshold , '\n')
        
        if error <   error_theshold:
            ### PASSED 
            print('PASSED\n\n')
            return True
        else:
            return False 
            
            
            

if __name__=='__main__':
    '''
    Testing 
    '''
   
    
    
    
    
    ################################################################
    ################################################################
    seed = 1211
    nspins = 9    
    num_layers =3
    num_itr =1
    machine_precision = 10  
    significant_figures = 3 
    eta = 100
    shots = 10**(2*significant_figures)
    Q = getRandomQ(nspins)
    
    
    inputs={}
    inputs['seed']=seed
    inputs['nspins']=nspins
    inputs['num_layers']=num_layers
    inputs['num_itr']=num_itr
    inputs['machine_precision']=machine_precision
    inputs['significant_figures']=significant_figures
    inputs['eta']=eta
    inputs['shots']=shots    
    inputs['Q']=Q
    
    testObj = TestAmplitude(**inputs)
    testObj.test1()
    # testObj.test2()

    testObj.test3()