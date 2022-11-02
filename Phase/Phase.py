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
    def __init__(self, nspins, df_ampl, df_ampl_bm, 
                 circ, Q, significant_digits, 
                 shots,
                 j_ref=None):
        #### 
        self.circ = circ ## the U_t circuit
        self.Q = Q # the Q Pauli string given in form of an array of 0,...,3
        self.significant_digits = significant_digits # significant_figures
        self.shots = shots # significant_figures
        
        if j_ref==None:
            #### choose the ref 
            j_ref = np.random.randint(0, 2**nspins)
            while j_ref in df_ampl.index:
                j_ref = np.random.randint(0, 2**nspins)
                
            self.nspins  = nspins
            self.df_ampl = df_ampl
            self.df_ampl_bm = df_ampl_bm ### to test and bm only
            self.j_ref = j_ref
        else:
            self.j_ref = j_ref
    
    def getImagPart(self,):
        
        ###
        return getImagPart(self.df_ampl,  
                self.df_ampl_bm,
                self.circ, 
                self.Q,
                self.significant_digits, 
                self.nspins,
                self.shots,
                self.j_ref)
    
    def getRealPart(self,):        
        ###
        return getRealPart(self.df_ampl,  
                self.df_ampl_bm,
                self.circ, 
                self.Q,
                self.significant_digits, 
                self.nspins,
                self.shots,
                self.j_ref)
    