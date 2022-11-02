#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:32:55 2022

@author: pejmanjouzdani
"""

from Amplitude.Amplitude import Amplitude
from Phase.Phase import Phase
from qiskit import QuantumCircuit

class Hamiltonin():
    def __init__(self, Qs, ws, nspins):
        self.Qs = Qs
        self.ws = ws
        self.nspins = nspins
    

class simulator():
    def __init__(self,T, n_H, delta, Qs, ws, nspins , shots_amplitude,
                 shots_phase ,  ):
        self.T= T
        self.n_H= n_H
        self.delta = delta 
        self.hamiltonian =Hamiltonin(Qs, ws, nspins)
        self.nspins = nspins
        self.circ = QuantumCircuit(nspins)
        self.shots_amplitude = shots_amplitude
        self.shots_phase = shots_phase
        
        
        
        
    def __call__(self,):
        T = self.T
        n_H = self.n_H
        delta = self.delta 
        ws = self.hamiltonian.ws
        circ = self.circ
        circ = self.circ
        
        for t in range(int(T/delta)):
            for k in range(n_H):
                delta_k = delta * ws[k]
                
                Amplitude( circ_uhu, shots, eta, significant_figures, machine_precision=10)
            
            
    
    