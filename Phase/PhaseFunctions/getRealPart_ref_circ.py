#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 12:02:09 2022

@author: pejmanjouzdani
"""

import numpy as np
from MultiQubitGate.functions import multiqubit
from BasicFunctions.functions import getBinary

def getRealPart_ref_circ(j_ref, j2,nspins,  circ_adj):
    
    #### T Gate        
    p_12_int = j2^j_ref                
    ## operator
    P_12 = getBinary(p_12_int, nspins).tolist()+[0] #bitstring array of p12
    mult_gate, op_count = multiqubit(P_12, np.pi/4) # turned into T gate
    circ_uhu_adj = circ_adj.compose( mult_gate ) #add to the circuit
    
    return circ_uhu_adj