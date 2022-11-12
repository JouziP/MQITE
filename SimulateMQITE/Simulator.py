#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:32:55 2022

@author: pejmanjouzdani
"""
import os

# from  SimulateMQITE.log_config import logger



from qiskit import QuantumCircuit


from Amplitude.amplitude import AmplitudeClass as Amplitude
from Phase.phase import Phase
from UpdateCircuit.UpdateCircuit import UpdateCircuit

from BasicFunctions.functions import  getUQUCirc, getQCirc






class Hamiltonin():
    def __init__(self, Qs, ws, nspins):
        self.Qs = Qs
        self.ws = ws
        self.nspins = nspins
    

class Simulator():
    def __init__(self,T, n_H, delta, Qs, ws, nspins , shots_amplitude,
                 shots_phase ,  eta, significant_figures, 
                 machine_precision, AmplTestLevel=0):
        
        
        
        
        
        
        
        # # proof check if the Values and Types are correct
        # self.checkValuesAndType(T, n_H, delta, Qs, ws, nspins , shots_amplitude,
        #          shots_phase ,  eta, significant_figures, 
        #          machine_precision)
        
        ### Static Attributes
        self.T= T
        self.n_H= n_H
        self.delta = delta 
        self.hamiltonian =Hamiltonin(Qs, ws, nspins)
        self.nspins = nspins
        self.shots_amplitude = shots_amplitude
        self.shots_phase = shots_phase
        self.eta = eta
        self.significant_figures= significant_figures
        self.machine_precision = machine_precision
        self.AmplTestLevel =AmplTestLevel
        ### Variable Attributes
        self.circ_U = QuantumCircuit(nspins)
        
        
        
        
    def __call__(self,):
        
        
        nspins              = self.nspins
        T                   = self.T
        n_H                 = self.n_H
        delta               = self.delta 
        ws                  = self.hamiltonian.ws
        Qs                  = self.hamiltonian.Qs
        
        shots_amplitude     = self.shots_amplitude
        shots_phase         = self.shots_phase
        eta = self.eta
        machine_precision   = self.machine_precision
        significant_figures = self.significant_figures
        
        #### initial circ_U
        circ_U              = self.circ_U
        
        for t in range(int(T/delta)):
            # logger.debug('\n=== === --> time step: %d \n'%( t) )
            
            for k in range(n_H):
                # logger.debug('=== === --> t: %d , k: %d \n'%( t, k) )
                
                print('t = ', t, ' ---  k= ' , k)
        
                Q = Qs[k]
                w = ws[k]                
                delta_k = delta * w
                
                circ_Q = getQCirc(QuantumCircuit.copy(circ_U), Q)                
                circ_UQU = getUQUCirc(QuantumCircuit.copy(circ_U), circ_Q)
                
                # ### constructor
                self.ampObj = Amplitude(circ_Q, circ_UQU, shots_amplitude, eta, significant_figures, machine_precision)
                
                # ### execution, test, save results, etc.
                # self.ampObj()
                
                # ### get phases
                self.phaseObj = Phase(self.ampObj, 
                                  nspins, 
                                  QuantumCircuit.copy(circ_U), # circ_U
                                  Q, 
                                  significant_figures, 
                                  shots_phase, 
                                  machine_precision)
                
                ### compute c_j^(r) and c_j^(im) for all j's in AmpObj
                self.phaseObj()
                
                ### get the parameters and associated j; y_j == [y_j^(r) , y_j^(i) ]
                [js, ys_j]=self.phaseObj.getY(delta_k)
                
                ### updateCircuit
                updateCircuitObj = UpdateCircuit()
                
                ### compute stuff
                circ_new, multigate_gate_stat =  updateCircuitObj(js, ys_j, QuantumCircuit.copy(circ_U),)
                
                ####
                circ_U = QuantumCircuit.copy(updateCircuitObj.circ_new) 
    
    
    @staticmethod
    def checkValuesAndType(T, n_H, delta, Qs, ws, nspins , shots_amplitude,
                 shots_phase ,  eta, significant_figures, 
                 machine_precision):        
        NotImplemented
                
      
if __name__=='__main__':
    
      
    

    import numpy as np
    seed = 12321    
    ########  seed random
    np.random.seed(seed)
    
    
    
    n_H = 2
    delta= 0.1
    T = 2 * delta
    Qs = [[1,2,1,0], [3,2,3,2]]
    ws = [0.2, 0.4]
    nspins = 4
    shots_amplitude = 1000
    shots_phase = 1000
    eta =100
    significant_figures = 3
    machine_precision  = 10
    AmplTestLevel=1
                
                
    simulator = Simulator(T, n_H, delta, Qs, ws, nspins , shots_amplitude,
                  shots_phase ,  eta, significant_figures, 
                  machine_precision, AmplTestLevel)
    simulator()
            
    
    