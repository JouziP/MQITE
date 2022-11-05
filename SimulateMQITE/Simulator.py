#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:32:55 2022

@author: pejmanjouzdani
"""
import os
import logging



from qiskit import QuantumCircuit


from Amplitude.Amplitude import Amplitude
from Phase.Phase import Phase
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
                 machine_precision):
        
        
        logging.info('%s.%s '%(self.__class__.__name__, self.__init__.__name__) )
        
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
        
        ### Variable Attributes
        self.circ = QuantumCircuit(nspins)
        
        
        
        
    def __call__(self,):
        logging.info('%s.%s '%(self.__class__.__name__, self.__call__.__name__) )
        
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
        
        #### initial circ
        circ                = self.circ
        
        for t in range(int(T/delta)):
            logging.info('\n=== === %s.%s --> time step: %d \n'%(self.__class__.__name__, self.__call__.__name__, t) )
            
            for k in range(n_H):
                logging.info('\n=== === %s.%s --> time step: %d --> k step %d \n'%(self.__class__.__name__, self.__call__.__name__, t, k) )

                Q = Qs[k]
                w = ws[k]                
                delta_k = delta * w
                
                circ_Q = getQCirc(circ, Q)                
                circ_UQU = getUQUCirc(circ, circ_Q)
                
                # ### constructor
                ampObj = Amplitude(circ_Q, circ_UQU, shots_amplitude, eta, significant_figures, machine_precision)
                
                ### execution, test, save results, etc.
                ampObj()
                
                # ### get phases
                phaseObj = Phase(ampObj, 
                                 nspins, 
                                 circ,
                                 Q, 
                                 significant_figures, 
                                 shots_phase, 
                                 machine_precision)
                
                ### compute c_j^(r) and c_j^(im) for all j's in AmpObj
                phaseObj()
                
        #         ### get the parameters and associated j; y_j == [y_j^(r) , y_j^(i) ]
        #         [js, ys_j]=phaseObj.getY(delta_k)
                
        #         ### updateCircuit
        #         updateCircuitObj = UpdateCircuit(ys_j, js, circ)
                
        #         ### compute stuff
        #         updateCircuitObj()
                
        #         ####
        #         circ = QuantumCircuit.copy(updateCircuitObj.circ_new) 
    
    
    @staticmethod
    def checkValuesAndType(T, n_H, delta, Qs, ws, nspins , shots_amplitude,
                 shots_phase ,  eta, significant_figures, 
                 machine_precision):
        NotImplemented
                
      
if __name__=='__main__':
    
    
    
    log_filename = 'Simulator.log'
    logging.basicConfig(filename=log_filename, 
                        format='%(asctime)s %(filename)s %(message)s', 
                        level=logging.INFO, 
                        force=True)
    
    
    
    import numpy as np
    seed = 12321    
    ########  seed random
    np.random.seed(seed)
    
    
    T = 1.
    n_H = 2
    delta= 0.1
    Qs = [[1,2,1,0], [3,2,3,2]]
    ws = [0.2, 0.4]
    nspins = 4
    shots_amplitude = 1000
    shots_phase = 1000
    eta =100
    significant_figures = 3
    machine_precision  = 10
                
                
                
    simulator = Simulator(T, n_H, delta, Qs, ws, nspins , shots_amplitude,
                  shots_phase ,  eta, significant_figures, 
                  machine_precision)
    simulator()
            
    
    