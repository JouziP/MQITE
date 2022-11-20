#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:33:48 2022

@author: pejmanjouzdani
"""



# from SimulateMQITE.log_config import logger

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit,  transpile
from qiskit import Aer


class AmplitudeClass:
    def __init__(self, circ_U, circ_UQU, Q, shots_amplitude, eta, significant_figures, machine_precision=10):
        self.Q = Q
        self.circ_U = circ_U
        self.circ_UQU = circ_UQU
        self.shots_amplitude = shots_amplitude
        self.eta = eta
        self.significant_figures = significant_figures
        self.machine_precision = machine_precision
        
        ############################ OUTPUTS
        #################### |cj|^2 and js upto significant digits 
        self.df_count = self.getIndexsFromExecute(self.circ_UQU, self.shots_amplitude)   
        ### keep eta of them -> {|c_j|} = df_ampl 
        ### the j's and c_j's from shots and observed bit string --> NO state vector
        self.df_ampl = self.getAmplitudes(
                                          pd.DataFrame.copy(self.df_count),
                                          self.eta
                                          )        
      
        
    @staticmethod
    def getIndexsFromExecute(circ_UQU, shots, backend = 'qasm_simulator'):
        pass
        '''
        appends classical wires to record measurements
        execute the circuit 'shots' time
        returns the observed bit-strings and distributions (counts)
        
        [AerSimulator('aer_simulator'),
         AerSimulator('aer_simulator_statevector'),
         AerSimulator('aer_simulator_density_matrix'),
         AerSimulator('aer_simulator_stabilizer'),
         AerSimulator('aer_simulator_matrix_product_state'),
         AerSimulator('aer_simulator_extended_stabilizer'),
         AerSimulator('aer_simulator_unitary'),
         AerSimulator('aer_simulator_superop'),
         QasmSimulator('qasm_simulator'),
         StatevectorSimulator('statevector_simulator'),
         UnitarySimulator('unitary_simulator'),
         PulseSimulator('pulse_simulator')]
        
        
        '''
        ### check if shots are type and value correct 
        if not isinstance(shots, int):
            raise TypeError('shots must be an integer')
            
        if shots<1 or shots>10**8:
            raise ValueError('Number of shots is either less than 1 or larger than 10^8')
       
        ### check if backend is correct    
        simulator = Aer.get_backend(backend)
       
        ### check the circ Value and Type
        
        ### Building measurment circuit
        num_qubits = circ_UQU.num_qubits
        circ_meas = QuantumCircuit(num_qubits, num_qubits)
        circ_meas = circ_meas.compose(circ_UQU)
        
        circ_meas.measure(
            [q for q in range(num_qubits)],
            [q for q in range(num_qubits)]
            )
        
        compiled_circuit = transpile(circ_meas, simulator)
            
        job = simulator.run(compiled_circuit, shots=shots)
        
        result = job.result()
        counts = result.get_counts(compiled_circuit)
        
        
        ### Using Pandas dataframe to organize
        df_count = pd.DataFrame([counts.int_outcomes()]).T   
        df_count.columns=['n_j']
        
        return df_count
        
    @staticmethod   
    def getAmplitudes(df_count,eta ): 
        '''
        
         return the |c_j| of the most largest amplitudes
     
        '''
        
        
        
        
        
        chi = df_count.sum()['n_j']
        
        df_count  = df_count/chi
        
        df_ampl = df_count.apply(lambda x: np.sqrt(x))
        df_ampl.columns=['|c_j|']
     
        return df_ampl
   
    
        
    


if __name__=='__main__':
    pass

    from qiskit import QuantumCircuit
   
   
    
    ################################################################
    ################################################################
    ##################### FOR TEST           #######################
    ################################################################
    ################################################################
    ################################################################    
    from BasicFunctions.getRandomQ import getRandomQ
    from BasicFunctions.getRandomU import getRandomU
    from BasicFunctions.getQCirc import getQCirc
    from BasicFunctions.getUQUCirc import getUQUCirc
    
   
    
    ################################################################
    ################################################################
    
    seed = 1253
    np.random.seed(seed)
    
    ################################################################
    ################################################################
    nspins = 12    
    num_layers =2
    num_itr =1
    machine_precision = 10        
    shots_amplitude = 100
    eta = 100
    significant_figures = 2#np.log10(np.sqrt(shots_amplitude)).astype(int)
    TestLevel =  0
    
    
    
    circ_U = getRandomU(nspins, num_layers)
    Q = getRandomQ(nspins)
    circ_UQ = getQCirc(circ_U, Q)
    circ_UQU = getUQUCirc(circ_U, circ_UQ)
    
    ################################# TEST
    ###### Constructor
    myAmplObj = AmplitudeClass( circ_U, circ_UQU, Q, shots_amplitude, eta, significant_figures, machine_precision=10)
    print(myAmplObj.df_count)
    print(myAmplObj.df_ampl)
    
    
    