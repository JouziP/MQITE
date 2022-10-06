#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:05:46 2022

@author: pejmanjouzdani
"""


      
from time import time    

import numpy as np
from numpy import linalg as lg
from scipy.sparse.linalg import cg
import pandas as pd
from matplotlib import pyplot as plt
from functools import wraps

from qiskit.extensions import IGate,XGate,YGate,ZGate
from qiskit import QuantumCircuit, execute, BasicAer
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import partial_trace, Statevector
from qiskit import Aer

class RydbergAtoms():
    def __init__(self,nspins, Rb, Omega):    
        ### nAtoms
        self.nspins= nspins
        ### Blockade Radius
        self.Rb = Rb
        #### Rabi Frequency coupling
        self.Omega = Omega
    def getLattice(self,):
        pass
            