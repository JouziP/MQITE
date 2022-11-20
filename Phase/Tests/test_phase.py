#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 09:57:37 2022

@author: pejmanjouzdani
"""



import pandas as pd
import numpy as np

from BasicFunctions.getQCirc import getQCirc
from BasicFunctions.getRandomQ import getRandomQ
from BasicFunctions.getRandomU import getRandomU
from BasicFunctions.getUQUCirc import getUQUCirc
from BasicFunctions.getState import getState




from Amplitude.amplitude import AmplitudeClass as Amplitude
from Phase.phase import Phase


class TestPhase:
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
        self.shots_amplitude = inputs['shots-amplitude']
        self.shots_phase = inputs['shots-phase']
        self.Q = inputs['Q']
    
    def test_real_m1(self, amplObj, phaseObj, nspins,  machine_precision=10):
        self.df_real_m1s_qc_vs_stvec = TestPhase.test_real_m1_static(amplObj, phaseObj, nspins,  machine_precision)
        return
    
    def test_ampl(self, amplObj,  machine_precision=10):
        self.df_ampl_sim_vs_stvec = TestPhase.test_ampl_static(amplObj,  machine_precision)
        return 
        
    @staticmethod 
    def test_ampl_static( amplObj,  machine_precision=10):
        '''
        tests if ampls are the same as the one implied from state vector (stvec)
        '''
        df_ampl = pd.DataFrame.copy(amplObj.df_ampl)
        df_ampl.columns=['|c_j|-qc']
        js = df_ampl.index.tolist()
        
        ### BENCHMARK AMPL
        state_vec= getState(amplObj.circ_UQU, machine_precision) 
        state_vec_js = state_vec[js, :]
        df_ampl_benchmark = pd.DataFrame([ np.sqrt(x[0,0]  *x[0,0].conjugate()).real for x in state_vec_js ])
        df_ampl_benchmark.columns=['|c_j|-stvec']
        df_ampl_benchmark.index=js
        
        df_ampl_sim_vs_stvec = pd.concat((df_ampl_benchmark, df_ampl), axis = 1)
        
        error = np.abs(df_ampl_sim_vs_stvec[df_ampl_sim_vs_stvec.columns[0]] - df_ampl_sim_vs_stvec[df_ampl_sim_vs_stvec.columns[1]])
        df_ampl_sim_vs_stvec['error'] = error
        
        return df_ampl_sim_vs_stvec
        
        
    @staticmethod
    def test_real_m1_static( amplObj, phaseObj, nspins, machine_precision=10):        
        '''
        The m1 from the real part circuit 
        
        '''
        
        df_ampl = pd.DataFrame.copy(amplObj.df_ampl)
        df_ampl.columns=['|c_j|-qc']
        js = df_ampl.index.tolist()
        
       
        circ_adj =  Phase.getRealPart_base_circ(nspins, amplObj.circ_U, amplObj.Q, phaseObj.j_ref)
        
        m1s_stvec = []
        for j2 in js:
            circ_uhu_adj = Phase.getRealPart_ref_circ(phaseObj.j_ref, j2, nspins,  circ_adj)
            state_vec_adj = getState(circ_uhu_adj, machine_precision) 
            m1 = state_vec_adj[phaseObj.j_ref, 0]
            m1s_stvec .append( (m1.conjugate() * m1 ).real )
            
        m1s_stvec = pd.DataFrame(m1s_stvec, columns=['m1-stvec'])
        m1s_stvec.index = js
        
        m1s_stvec =  m1s_stvec
       
        ### parts_real_m1s
        phaseObj.getRealPart() 
       
        df_m1s_qc_vs_stvec = pd.concat((m1s_stvec, phaseObj.parts_real_m1s), axis=1)
        error = np.abs(df_m1s_qc_vs_stvec[df_m1s_qc_vs_stvec.columns[0]] - df_m1s_qc_vs_stvec[df_m1s_qc_vs_stvec.columns[1]])
        df_m1s_qc_vs_stvec['error'] = error
        
        df_m1s_qc_vs_stvec = df_m1s_qc_vs_stvec
        
        return df_m1s_qc_vs_stvec
    
    
    @staticmethod
    def test_real_part_static( amplObj, phaseObj, gamma, nspins, significant_figures,  machine_precision=10):   
        
        
        
        ### from stvec
        df_m1s_qc_vs_stvec = TestPhase.test_real_m1_static(amplObj, phaseObj, nspins)        
        df_ampl_sim_vs_stvec = TestPhase.test_ampl_static(amplObj,  machine_precision)
        
        c2_stvec = df_ampl_sim_vs_stvec['|c_j|-stvec'].values.tolist()
        m1s_stvec = df_m1s_qc_vs_stvec['m1-stvec'].values.tolist()
             
        real_part = lambda m_ref, c2 : (np.round(m_ref, significant_figures) - (1/4) * np.round(c2**2 , significant_figures)*\
                         (np.cos(gamma/2)**2) - (1/4)*(np.sin(gamma/2))**2 )/\
                           ((-1/2) * np.cos(gamma/2) * np.sin(gamma/2)) 
                           
                     
        
        real_parts_stvec = pd.DataFrame(list(map(real_part, c2_stvec, m1s_stvec)))
        real_parts_stvec.columns=['c_real_stvec']
        
        real_parts_stvec.index = df_m1s_qc_vs_stvec.index
        
        
        ##### sim
        phaseObj.gamma = gamma
        phaseObj.getRealPart()
        
        ################
        
        ####  exact
        js = df_m1s_qc_vs_stvec.index
        state_vec= getState(amplObj.circ_UQU, machine_precision) 
        state_vec_js = state_vec[js, :]
        real_parts_exact = pd.DataFrame(state_vec_js.real.T.tolist()[0], columns = ['c_real_exact'], index=js)
        
        df = pd.concat((phaseObj.parts_real, real_parts_stvec, real_parts_exact), axis=1)
        
        df['error'] = np.abs(df['c_real_sim'] - df['c_real_exact'])
        
        return df
        
        
        
        
    
    
    
    
   
        
if __name__=='__main__':
    '''
    Testing 
    '''
   
    
    
    
    
    ################################################################
    ################################################################
    seed = 1211
   
    
    
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
    gamma= np.pi/4
    
    
    
    
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
    
    testPhase = TestPhase(**inputs)
    circ_U = getRandomU(nspins, num_layers) 
    circ_Q = getQCirc(circ_U, Q)
    circ_UQU = getUQUCirc(circ_U, circ_Q)
    
    amplObj  = Amplitude(circ_U, circ_UQU, Q, shots_amplitude, eta, significant_figures)
    phaseObj = Phase( amplObj.df_ampl.copy(), nspins, amplObj.circ_U, amplObj.Q, 
                          significant_figures, shots_phase)
    
    
    df_ampl = amplObj.df_ampl
    
    #### Tests     amplObj
    testPhase.test_ampl(amplObj,  machine_precision)
    print(testPhase.df_ampl_sim_vs_stvec.round(significant_figures))
    print()
    
    #### Tests real_m1s from qc vs m1 from stvec
    testPhase.test_real_m1(amplObj, phaseObj, nspins,  machine_precision)
    print(testPhase.df_real_m1s_qc_vs_stvec.round(significant_figures))
    print()
    
    
    
    df = testPhase.test_real_part_static(amplObj, phaseObj, gamma, nspins, significant_figures)
    
    print(df.round(significant_figures))