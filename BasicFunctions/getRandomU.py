#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 12:15:52 2022

@author: pejmanjouzdani
"""

import numpy as np
from qiskit import QuantumCircuit


################################################################
################################################################
def getRandomU(nspins, num_layers=10):
    circ_U = QuantumCircuit(nspins)
    for l in range(num_layers):
        for i in range(nspins):
            ##############
            q=np.random.randint(nspins)
            g=np.random.randint(1, 4)
            p=np.random.uniform(-1,1)
            if g==1:
                circ_U.rx(p,q)
            if g==2:
                circ_U.ry(p,q)
            if g==2:
                circ_U.rz(p,q)
            ##############
            q=np.random.randint(nspins-1)
            circ_U.cnot(q, q+1)
    return circ_U     