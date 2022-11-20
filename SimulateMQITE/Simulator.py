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
from Observables.observables import Observables
from Benchmarks.amplitudeBenchmark import AmplitudeBenchmark


from BasicFunctions.getUQUCirc import  getUQUCirc
from BasicFunctions.getQCirc import  getQCirc







class Hamiltonin():
    def __init__(self, Qs, ws, nspins):
        self.Qs = Qs
        self.ws = ws
        self.nspins = nspins
    

class Simulator():
    def __init__(self,T, n_H, delta, Qs, ws, nspins , shots_amplitude,
                 shots_phase ,  eta, significant_figures, 
                 machine_precision, observable_matrix=None,
                 observable_exact = None):
        
        
        
        
        
        
        
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
        self.observable_matrix =observable_matrix
        self.observable_exact = observable_exact
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
        
        #### initial expectaion   
        self.observableObj = Observables(self.observable_matrix, 
                                         self.machine_precision,
                                         self.observable_exact)
        self.observableObj.getExpectation_matrixVersion(
                                                 circ_U, t=0 , k=0)
        
        #### initialize Benchmark         
        self.amplBenchmarkObj =  AmplitudeBenchmark()
        
        
        for t in range(int(T/delta)):
            # logger.debug('\n=== === --> time step: %d \n'%( t) )
            
            for k in range(n_H):
                # logger.debug('=== === --> t: %d , k: %d \n'%( t, k) )
                print('\n\n\n\n')
                print('=================================================')
                print('=================================================')
                print('=================================================')
                print('===================== FROM SIMULATION :')
                print('t = ', t, ' ---  k= ' , k)
                print()
        
                Q = Qs[k]
                w = ws[k]                
                delta_k = delta * w
                
                circ_Q = getQCirc(QuantumCircuit.copy(circ_U), Q)                
                circ_UQU = getUQUCirc(QuantumCircuit.copy(circ_U), circ_Q)
                
                ##################################################  Amplitude
                # executes the UQU circuit and obtains the |c_j| and {|j>}
                self.ampObj = Amplitude(circ_Q, circ_UQU, shots_amplitude, 
                                eta, significant_figures, machine_precision)
                
                #####Benchmark                
                self.amplBenchmarkObj.getAmplFromStateVector(self.ampObj, t, k)
                self.amplBenchmarkObj.display()
                ################################################## Phases
                # prepare inputs
                self.phaseObj = Phase(self.ampObj.df_ampl.copy(), # df_ampl
                                      nspins, 
                                      circ_U.copy(), # circ_U
                                      Q, 
                                      significant_figures, 
                                      shots_phase, 
                                      machine_precision)
                
                # compute c_j^(r) and c_j^(im) for all j's in AmpObj
                self.phaseObj()
                
                # get delta * cj values, and corresponding {|j>}
                [js, delta_times_cj]=self.phaseObj.getY(delta_k)
                
                ############################################## updateCircuit
                updateCircuitObj = UpdateCircuit()
                
                # uses the delta * cj to create a new layer of cicuits
                # and adds to current circ_U
                # the result is updateCircuitObj attribute circ_new
                updateCircuitObj(js, delta_times_cj, circ_U.copy() )
                
                # copy circ_new into current circ_U
                circ_U = QuantumCircuit.copy(updateCircuitObj.circ_new) 
    
                # compute observable
                self.observableObj.getExpectation_matrixVersion(circ_U,
                                                           t, k)
                self.observableObj.display()
                # go to the next k and Q_k
            
            # print('\n============================ ')
            
            # print('\nt = ', t, ' ---  k= ' , k, '--- E_expect = ', self.observableObj.current_expectation)
            # print(  't = ', t, ' ---  k= ' , k, '--- len(js)  = ', len(js), '\n')
            # print(  't = ', t, ' ---  k= ' , k, '--- amplError  = ', self.amplBenchmarkObj.current_error, '\n')
            # print('\n============================ ')
            ###############
            # go t the next imaginary time step t
                
    @staticmethod
    def checkValuesAndType(T, n_H, delta, Qs, ws, nspins , shots_amplitude,
                 shots_phase ,  eta, significant_figures, 
                 machine_precision):        
        NotImplemented
                
      
if __name__=='__main__':
    
      
    
    import pandas as pd
    import numpy as np
    seed = 12321    
    ########  seed random
    np.random.seed(seed)
    
    
    
    n_H = 2
    delta= 0.2
    T =20 * delta
    Qs = [
        [1, 1, 1, 0, 1, 2], 
        [3, 2, 3, 2, 3, 1],
          ]
    ws = [0.2, 0.4]
    nspins = len(Qs[0])
    shots_amplitude = 10000
    shots_phase = 10000
    eta =100
    significant_figures = 3
    machine_precision  = 10
    
         

        
    inputs={}
    inputs['seed']=seed
    inputs['nspins']=nspins        
    inputs['T']=T
    inputs['machine_precision']=machine_precision
    inputs['significant_figures']=significant_figures
    inputs['eta']=eta
    inputs['shots-amplitude']=shots_amplitude    
    inputs['shots-phase']=shots_phase    
    inputs['delta']=delta
    inputs['n_H']=n_H    
    
    
    print(pd.DataFrame([inputs], index=['input values']).T , '\n')
    
    
    
    #### Exact value
    from BasicFunctions.exact_calculations import getExact
    
    E_exact, _, H_mtx, ___ = getExact(Qs, ws)
    
    print('E exact = ' , E_exact)
    
    simulator = Simulator(T, n_H, delta, Qs, ws, nspins , shots_amplitude,
                  shots_phase ,  eta, significant_figures, 
                  machine_precision, 
                  observable_matrix=H_mtx,
                  observable_exact=E_exact)
    simulator()
    print(pd.DataFrame([inputs], index=['input values']).T , '\n')
    print('E exact = ' , E_exact)
            
    
    