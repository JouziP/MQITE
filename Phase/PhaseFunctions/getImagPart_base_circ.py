#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 11:35:58 2022

@author: pejmanjouzdani
"""

import numpy as np

from qiskit import QuantumCircuit

from BasicFunctions.functions import getBinary


def getImagPart_base_circ(nspins, circ_U , Q, j_ref,  gamma =   np.pi/10):
    
    circ_adj = QuantumCircuit(nspins+1)
    
    circ_adj.ry(gamma, qubit=-1)  ### R_gamma
    circ_adj.x(qubit=-1)  ### X
    
    ### U
    ### attaches U to the q=1 ... q=n qubits, while q=0 is the ancillary 
    circ_adj = circ_adj.compose(QuantumCircuit.copy(circ_U) ) 
    
    ### control-Q ; Ancillary - n target 
    for (q,o) in enumerate(Q):
        if o==1:            
            circ_adj.cx(-1, q)
        if o==2:
            circ_adj.cy(-1, q)
        if o==3:
            circ_adj.cz(-1, q)
    
    ### U^    
    circ_adj = circ_adj.compose(QuantumCircuit.copy(circ_U).inverse())
    
    ### control-P_{0 j_ref}
    circ_adj.x(qubit=nspins)    
    J1 = list(getBinary(j_ref, nspins)) + [0]
    
    for (q,o) in enumerate(J1):
        if o==1:            
            circ_adj.cx(nspins, q)                           
    circ_adj.x(nspins)  
    
    
    ### H on ancillary
    circ_adj.h(nspins)
    
    return circ_adj