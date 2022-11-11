#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:37:58 2022

@author: pejmanjouzdani
"""

from qiskit.quantum_info import Statevector
import numpy as np

def getState(circ, machine_precision):
    return np.matrix(Statevector(circ)).reshape(2**circ.num_qubits,1).round(machine_precision)