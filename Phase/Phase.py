#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:40:38 2022

@author: pejmanjouzdani
"""



import pandas as pd
import numpy as np


from Phase.PhaseFunctions.getImagPart import getImagPart
from Phase.PhaseFunctions.getRealPart import getRealPart


class Phase:
    def __init__(self, 
                 amplObj,
                 nspins, 
                 circ, 
                 Q,
                 significant_figure, 
                 shots,
                 machine_precision=10,
                 j_ref=None):
        
        #### static
        self.amplObj            = amplObj
        self.nspins             = nspins
        self.circ               = circ ## the U_t circuit
        self.Q                  = Q # the Q Pauli string given in form of an array of 0,...,3
        self.significant_figure = significant_figure # significant_figures
        self.shots              = shots # 
        self.machine_precision  = machine_precision        
        self.df_ampl            = amplObj.df_ampl# |c_j| in U^QU|0>=\sum_j c_j|j>
        
        
        if j_ref==None:
            #### choose the ref 
            j_ref = np.random.randint(0, 2**nspins)
            while j_ref in self.df_ampl.index:
                j_ref = np.random.randint(0, 2**nspins)            
        else:
            pass
        self.j_ref = j_ref
            
    def __call__(self, TestLevel=0):
        if TestLevel==0:
            self.parts_imag = self.getImagPart()
            self.parts_real = self.getRealPart()
            return 
        
        if TestLevel==1:
            self.parts_imag = self.getImagPart()
            self.parts_real = self.getRealPart()
            ### test if the parts real is the same as in the state_vector
            
            
            
            return 
        
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
        
    
    def getImagPart(self,):
        
        ###
        return getImagPart(self.df_ampl,  
                           self.circ, 
                           self.amplObj.circ_UQU, 
                           self.Q,
                           self.significant_figure, 
                           self.nspins,
                           self.shots,
                           self.j_ref,
                           self.machine_precision)
    
    def getRealPart(self,):        
        ###
        return getRealPart(self.df_ampl,                  
                           self.circ, 
                           self.amplObj.circ_UQU, 
                           self.Q,
                           self.significant_figure, 
                           self.nspins,
                           self.shots,
                           self.j_ref,
                           self.machine_precision)
    