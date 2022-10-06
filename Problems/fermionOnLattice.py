#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:26:06 2022

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

class fermionOn2DLattice():
    def __init__(self,                 
                 nfermion,
                 Lx,
                 Ly,
                 t_hop,
                 U,
                 ):        
        self.nfermion = nfermion
        self.Lx = Lx
        self.Ly = Ly
        self.t_hop = t_hop
        self.U =U
        
        self.getNeighbs()
        self.nspins = len(self.odd_lattice.keys()) + len(self.even_lattice.keys()) 
    
    
    def getNeighbs(self):
        
        Lx = self.Lx
        Ly = self.Ly
        
        ### for spin
        nx = 2*Lx
        ny = 2*Ly
        
        even_lattice = {}
        odd_lattice = {}
        
        for ix in range(0,Lx):            
            for iy in range(0, Ly):
                ###############     X direction
                
                #### Bulk
                if ix != Lx-1: 
                    ######  EVEN
                    n1 = ix*2 + iy*2*Lx
                    n2 = (ix+1)*2 + iy*2*Lx                         
                    if str(n1) in even_lattice.keys():
                        even_lattice[str(n1)].append(n2)
                    else:
                        even_lattice[str(n1)]= [n2] 
                    ######  Odd
                    n1 = ix*2 +1 + iy*2*Lx
                    n2 = (ix+1)*2+1 + iy*2*Lx                                               
                    if str(n1) in odd_lattice.keys():
                        odd_lattice[str(n1)].append(n2)
                    else:
                        odd_lattice[str(n1)]= [n2]
                    
                                
                #### Boundary    
                else:
                    ######  EVEN
                    n1 = ix*2 + iy*2*Lx
                    n2 = (0)*2 + iy*2*Lx                         
                    if str(n1) in even_lattice.keys():
                        even_lattice[str(n1)].append(n2)
                    else:
                        even_lattice[str(n1)]= [n2]                    
                    ######  Odd
                    n1 = ix*2 +1 + iy*2*Lx
                    n2 = (0)*2+1 + iy*2*Lx                           
                    if str(n1) in odd_lattice.keys():
                        odd_lattice[str(n1)].append(n2)
                    else:
                        odd_lattice[str(n1)]= [n2]
                    
                ###############     Y direction
                ###############                                    
                #### Bulk
                if iy != Ly-1: 
                    ######  EVEN
                    n1 = ix*2 + iy*2*Lx
                    n2 = (ix)*2 + (iy+1)*2*Lx                                        
                    if str(n1) in even_lattice.keys():
                        even_lattice[str(n1)].append(n2)
                    else:
                        even_lattice[str(n1)]= [n2]
                    ######  Odd
                    n1 = ix*2 +1 + iy*2*Lx
                    n2 = (ix)*2+1 + (iy+1)*2*Lx                                        
                    if str(n1) in odd_lattice.keys():
                        odd_lattice[str(n1)].append(n2)
                    else:
                        odd_lattice[str(n1)]= [n2]
                
                #### Boundary         
                else:
                    ######  EVEN
                    n1 = ix*2 + iy*2*Lx
                    n2 = (ix)*2 + 0*2*Lx                                        
                    if str(n1) in even_lattice.keys():
                        even_lattice[str(n1)].append(n2)
                    else:
                        even_lattice[str(n1)]= [n2]
                    ######  Odd                  
                    n1 = ix*2 +1 + iy*2*Lx
                    n2 = (ix)*2+1 + 0*2*Lx                                                            
                    if str(n1) in odd_lattice.keys():
                        odd_lattice[str(n1)].append(n2)
                    else:
                        odd_lattice[str(n1)]= [n2]
                    
                    
                    
        self.even_lattice = even_lattice
        self.odd_lattice = odd_lattice




if __name__=='__main__':
    pass    
    nfermion =3
    Lx= 3
    Ly=3
    t_hop=1
    U=1
    
    fm = fermionOn2DLattice(
                       nfermion,
                       Lx,
                       Ly,
                       t_hop,
                       U)
    
    fm.getNeighbs()
        
    