#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:21:51 2022

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
     
   
    
def test1(phaseObj):
    indexs = phaseObj.df_ampl.index.tolist()
    print(indexs)
    
    circ_adj_for_real_part = phaseObj.getRealPart_base_circ(phaseObj.nspins, phaseObj.circ, phaseObj.Q, phaseObj.j_ref, phaseObj.gamma)    
    m1s          = phaseObj.getMsReal(indexs, phaseObj.j_ref, phaseObj.nspins, circ_adj_for_real_part, phaseObj.shots, phaseObj.significant_figures )
    print('m1s real ' , m1s)
    
    
    
    # phaseObj.gamma = 0.1
    circ_adj_for_imag_part = phaseObj.getImagPart_base_circ(phaseObj.nspins, phaseObj.circ, phaseObj.Q, phaseObj.j_ref, phaseObj.gamma)    
    m1s          = phaseObj.getMsImag(indexs, phaseObj.j_ref, phaseObj.nspins, circ_adj_for_imag_part, phaseObj.shots, phaseObj.significant_figures )
    print('m1s imag ' , m1s)
    
    
def test2(phaseObj, amplObj):
    indexs = phaseObj.df_ampl.index.tolist()
    
    ######### Real Parts    
    phaseObj.getRealPart()
    c_real =  phaseObj.c_real
    m1s_from_real =  phaseObj.m1s_from_real
    c2_2power2__real =  phaseObj.c2_2power2__real
    
    ######### imag Parts  
    phaseObj.getImagPart()
    c_imag =  phaseObj.c_imag
    m1s_from_imag =  phaseObj.m1s_from_imag
    c2_2power2__imag =  phaseObj.c2_2power2__imag
    
    #####
    c = pd.DataFrame(c_real.values + c_imag.values*1j).round(phaseObj.significant_figures)    
    c.columns=['c_sim']
    
    ##### exact 
    state_vec_UQU = getState(amplObj.circ_UQU, amplObj.machine_precision)
    c['c_stateVec_UQU'] = pd.DataFrame(\
       state_vec_UQU[indexs, :].real.round(phaseObj.significant_figures) +\
       1j*state_vec_UQU[indexs, :].imag.round(phaseObj.significant_figures))
       
    c.index = c_real.index
    c = c.sort_index()
    # print(c.round(phaseObj.significant_figures))
    # print()
    
    c['error'] = c['c_sim'] - c['c_stateVec_UQU'] 
    
    error_real = sum([x*x for x in np.real(c.error)])
    error_real = np.sqrt(error_real)/len(c.index)
    
    error_imag = sum([x*x for x in np.imag(c.error)])
    error_imag = np.sqrt(error_imag)/len(c.index)
    
    print('Test2 on phaseObj and amplObj : ' )
    print('indexs =  : ' , np.array(c.index))
    print('error_real =  : ' ,  np.round(error_real, phaseObj.significant_figures))
    print('error_imag =  : ' ,  np.round(error_imag, phaseObj.significant_figures))
    
    
    
if __name__=='__main__':
    '''
    Testing 
    '''
   
    
    
    
    
    ################################################################
    ################################################################
    seed = 1211
   
    np.random.seed(seed)
    
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
    
    
    print(pd.DataFrame([inputs], index=['input values']).T , '\n')
    
    circ_U = getRandomU(nspins, num_layers) 
    circ_Q = getQCirc(circ_U, Q)
    circ_UQU = getUQUCirc(circ_U, circ_Q)
    
    amplObj  = Amplitude(circ_U, circ_UQU, Q, shots_amplitude, eta, significant_figures)
    
    phaseObj = Phase( amplObj.df_ampl.copy(), nspins, amplObj.circ_U, amplObj.Q, 
                          significant_figures, shots_phase)
    
    
    
    ############## Test
    test2(phaseObj, amplObj)
   