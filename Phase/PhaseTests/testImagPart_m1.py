#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 09:57:37 2022

@author: pejmanjouzdani
"""
import numpy as np

from BasicFunctions.functions import getBinary, getState


def test1(circ_uhu_adj, significant_digits, j_ref, j2,  m1):
    print('-----   START test1')
    print('j  = ', j2)
    ############################################
    ## M1 WITH THE RESULTS FROM CIRCUIT EXECUTION 
    print('m1 from shots = ', m1)
    
    #####  get components from the state vector
    # to be replaced by shots
    circ_state_adj  = getState(circ_uhu_adj, significant_digits)                                              
    # ancila == 0
    circ_state_adj = circ_state_adj.reshape(2, circ_state_adj.shape[0]//2).T[:, 0]        
    m1_exact = circ_state_adj[j_ref, 0]
    
    m1_exact = (m1_exact*m1_exact.conjugate()).real.round(significant_digits)
    print('m1 from state-vector (THIS IS FOR BM) = ', m1_exact)
    ############################################
    print('-----   END test1')
    print()
    
    
    

def test2(df_comp_exact, j2,significant_digits, c2_2):
    print('-----   START test2')
    print('j  = ', j2)
    
    ## C**2 FROM  CIRCUIT EXECUTION 
    print('c2_2 from shots = ', c2_2)
    
    ############################################
    c2_2_exact = df_comp_exact[0][j2].real**2 + df_comp_exact[0][j2].imag**2
    c2_2_exact = np.round(c2_2_exact , significant_digits)
    print('c2_2_exact from state-vector (THIS IS FOR BM) = ', c2_2_exact)
    ############################################
    
    
    print('-----   END test2')
    print()