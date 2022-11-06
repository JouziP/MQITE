#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 12:18:27 2022

@author: pejmanjouzdani
"""

import numpy as np
from qiskit.circuit.exceptions import CircuitError
from qiskit import QuantumCircuit

def getQCirc(circ_U, Q):    
    #############
    # Qiskit error
    if not isinstance(circ_U, QuantumCircuit):
        raise CircuitError('The circuit is not an instance of the QuantumCircuit')
    
    # TypeError
    if not isinstance(Q, np.ndarray):
        if not isinstance(Q, list):
            raise TypeError(' the provided Pauli string operator is not am instance of list or numpy array ')
    #############

    
    circ_Q = QuantumCircuit.copy(circ_U)

    #############
    # ValueError
    ### the number of qubits must be the same as number of operators in Q
    if not circ_Q.num_qubits == len(Q):
        raise ValueError('the number of qubits in the circuit is not the same as the  number of Pauli operators in Q')
    #############
    
    
    for (q, o) in enumerate(Q):
        if o==0:
            pass
        if o==1:
            circ_Q.x(q)
        if o==2:
            circ_Q.y(q)
        if o==3:
            circ_Q.z(q)
            
    return circ_Q