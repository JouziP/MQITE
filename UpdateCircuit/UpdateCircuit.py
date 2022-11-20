#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:12:30 2022

@author: pejmanjouzdani
"""



import numpy as np
import pandas as pd

from qiskit import QuantumCircuit





from BasicFunctions.functions import timing, getBinary, getGateStats
from MultiQubitGate.functions import multiqubit


class UpdateCircuit:
    def __init__(self,):
        pass
    
    
    def __call__(self, list_j, list_y_j, circ,):
        self.circ_new, self.multigate_gate_stat = self.updateCirc(list_j, list_y_j, circ)

        # return self.circ_new, self.multigate_gate_stat


    @staticmethod
    def updateCirc(list_j, list_y_j, circ,):
    
        nspins = circ.num_qubits
        
        circ_P = QuantumCircuit(nspins)
        
        multigate_gate_stat = pd.DataFrame()
        for im in range(len(list_j)):
            # print('\n8888888888888888 ')
            ####### imag
            m=list_j[im]
            h = getBinary(m, nspins)   
            
            ### imag part  
            y_j= list_y_j[im].imag  
            
            if np.round(y_j, 4)==0:
                pass
            else:            
                mult_gate, op_count = multiqubit(h, y_j)            
                circ_P = circ_P.compose(mult_gate)
            
        
            ####### REAL
            ### replace 3 1 with Y         
            y_j=  -list_y_j[im].real        
            h[np.nonzero(h)[0][0]]=2     
            
            if np.round(y_j, 4)==0:
                pass
            else:
                mult_gate, op_count =multiqubit(h, y_j)
                ####### statistics
                multigate_gate_stat = pd.concat((multigate_gate_stat,
                                                 pd.DataFrame([op_count], index=[m]))  )        
                
                circ_P = circ_P.compose(mult_gate)
        
        circ_new = circ_P.compose(circ)
        
        
        return circ_new, multigate_gate_stat