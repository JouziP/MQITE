#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 12:26:49 2022

@author: pejmanjouzdani
"""


from qiskit import QuantumCircuit

def getUQUCirc(circ_U, circ_Q):
    circ_UQU = QuantumCircuit.copy(circ_Q)  ## QU|0>
    circ_UQU = circ_UQU.compose(QuantumCircuit.copy(circ_U.inverse()) )## U^QU|0>
    return circ_UQU ## U^QU|0>