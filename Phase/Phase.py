#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:40:38 2022

@author: pejmanjouzdani
"""

# # from SimulateMQITE.log_config import logger

import pandas as pd
import numpy as np


# from Phase.PhaseFunctions.getImagPart import getImagPart
# # from Phase.PhaseFunctions.getRealPart import getRealPart



from Phase.PhaseFunctions.computeAmplFromShots import computeAmplFromShots


from Phase.PhaseFunctions.getImagPart_base_circ import getImagPart_base_circ
from Phase.PhaseFunctions.getImagPart_ref_circ import getImagPart_ref_circ

from Phase.PhaseFunctions.getRealPart_base_circ import getRealPart_base_circ
from Phase.PhaseFunctions.getRealPart_ref_circ import getRealPart_ref_circ

class Phase:
    ##### CONSTRUCTOR
    def __init__(self, 
                  amplObj,
                  nspins, 
                  circ, 
                  Q,
                  significant_figure, 
                  shots,
                  machine_precision=10,
                  j_ref=None,
                  gamma= np.pi/10):
        
        #### static
        self.amplObj            = amplObj
        self.nspins             = nspins
        self.circ               = circ ## the U_t circuit
        self.Q                  = Q # the Q Pauli string given in form of an array of 0,...,3
        self.significant_figure = significant_figure # significant_figures
        self.shots              = shots # 
        self.machine_precision  = machine_precision        
        self.df_ampl            = amplObj.df_ampl# |c_j| in U^QU|0>=\sum_j c_j|j>
        self.gamma= gamma
        
        if j_ref==None:
            #### choose the ref 
            j_ref = np.random.randint(0, 2**nspins)
            while j_ref in self.df_ampl.index:
                j_ref = np.random.randint(0, 2**nspins)            
        else:
            pass
        self.j_ref = j_ref
    

    
    ##### OVERWRITING () OPERATOR     
    def __call__(self, TestLevel=0):
        
        ### PURE Q.COMPUTE
        if TestLevel==0:
            self.parts_imag = self.getImagPart()
            self.parts_real = self.getRealPart()
            return 
        
        ### Q.COMPUTE AND STATE-VECTOR BENCHMARK
        if TestLevel==1:
            
            self.parts_imag = self.getImagPart()
            self.parts_real = self.getRealPart()
            ### test if the parts real is the same as in the state_vector            
            return 
        
        ### Q.COMPUTE AND STATE-VECTOR BENCHMARK
        if TestLevel==2:
            
            
            pass # different scenario: the amplitude is from state-vector etc.
            
            
        
    ##### TURN C_J TO Y_J
    def getY(self, delta):
        '''
        uses the c_j^(r) and imag to compute the y_j_r and imaginary
        '''
        NotImplemented
        
        try:
            ## <0| U^ Q U |0>
            Q_expect = self.df_ampl.loc[0][0].real
        except:
            Q_expect = 0 
        
        ### norm    
        norm = 1- 2*delta * Q_expect + delta ** 2
        
        ### y_j
        df_y_j =  ((self.df_ampl)[0] * -(delta/norm) )                        
        df_y_j =  pd.DataFrame(df_y_j, columns=[0])
        
        ### j_list and y_list
        j_list = df_y_j.loc[df_y_j[0]!=0].index
        df_y_j = df_y_j.loc[j_list]            
        # remove 0 double check
        try:
            df_y_j = df_y_j.drop(index=0)
        except:
            pass            
        self.ys_j = df_y_j[0].tolist() 
        self.js = df_y_j.index.tolist()
        
        return [self.js, self.ys_j]
        
    
    
    ##### Q. COMPUTE  IMAG PART OF C_J
    def getImagPart(self,):
        
        circ_adj = getImagPart_base_circ(self.nspins, self.circ, self.Q, self.j_ref, self.gamma)
        
        #################### for each j2  The T_{j_ref -> j2} is different 
        indexs = self.df_ampl.index  ### the observed bit strings from shots; j's
        parts_imag= [[ 0 ]]  # ref has c_j_ref = 0 + 0j
        part_indexs=[self.j_ref]
        
        for j2 in indexs:
            circ_uhu_adj = getImagPart_ref_circ(self.j_ref, j2, self.nspins,  circ_adj)
                        
            #####  amplitude of m_ref from shots 
            m_ref, __ = computeAmplFromShots(circ_uhu_adj, self.shots, self.j_ref)
            m_ref = np.round(m_ref, self.significant_figure)
            
            
            #### amplitude  from shots
            c2_2 = self.df_ampl[0][j2]**2 ### |c_j|^2
            c2_2    = np.round(c2_2, self.significant_figure)
            
            #### compute the sin of theta        
            imag_part = (m_ref - (1/4) * c2_2**2 *\
                          (np.cos(self.gamma/2)**2) - (1/4)*(np.sin(self.gamma/2))**2 )/\
                            ((-1/2) * np.cos(self.gamma/2) * np.sin(self.gamma/2)) 
        
            #### round to allowed prcision
            imag_part = np.round(imag_part, self.significant_figure)
            
            ### collect results
            parts_imag.append([ imag_part ])
            part_indexs.append(j2)
        
        ### final results
        parts_imag = pd.DataFrame(parts_imag, index= part_indexs).round(self.significant_figure)
        parts_imag.columns=[ 'c_imag_sim' ]    
    
    
    
    
    ##### Q. COMPUTE  REAL PART OF C_J
    def getRealPart(self,):        
        
        circ_adj = getRealPart_base_circ(self.nspins, self.circ, self.Q, self.j_ref, self.gamma)
        
        #################### for each j2  The T_{j_ref -> j2} is different 
        indexs = self.df_ampl.index  ### the observed bit strings from shots; j's
        parts_real= [[ 0]]  # ref has c_j_ref = 0 + 0j
        part_indexs=[self.j_ref]
        
        for j2 in indexs:
            circ_uhu_adj = getRealPart_ref_circ(self.j_ref, j2, self.nspins,  circ_adj)
            #####  from shots
            m1, __ = computeAmplFromShots(circ_uhu_adj, self.shots, self.j_ref)
            m1 = np.round(m1, self.significant_figure)
            
            
           
                           
            #### amplitude  from shots
            c2_2 = self.df_ampl[0][j2]**2 ### |c_j|^2
            c2_2    = np.round(self.c2_2, self.significant_figure)
            
            
            
            
            #### compute the cos of theta        
            real_part = (m1 - (1/4) * c2_2**2 *\
                          (np.cos(self.gamma/2)**2) - (1/4)*(np.sin(self.gamma/2))**2 )/\
                            ((-1/2) * np.cos(self.gamma/2) * np.sin(self.gamma/2))
            
            #### round to allowed prcision
            real_part = np.round(real_part, self.significant_figure)
               
            ### collect results
            parts_real.append([ real_part ])
            part_indexs.append(j2)
           
        ### final results
        parts_real = pd.DataFrame(parts_real, index= part_indexs).round(self.significant_figure)        
        parts_real.columns=[ 'c_real_sim' ]    
        