#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:37:01 2022

@author: pejmanjouzdani
"""



import numpy as np
import pandas as pd


from qiskit import QuantumCircuit,  transpile
from qiskit import Aer


def getIndexsFromExecute(circ, shots, backend = 'qasm_simulator'):
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
    num_qubits = circ.num_qubits
    circ_meas = QuantumCircuit(num_qubits, num_qubits)
    circ_meas = circ_meas.compose(circ)
    
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





if __name__=='__main__':
    pass
    from qiskit import QuantumCircuit

    ################################################################
    ################################################################
    ##################### FOR TEST           #######################
    ################################################################
    ################################################################
    ################################################################

    ################################################################
    ################################################################
    def getRandomU(nspins, num_layers=10):
        circ = QuantumCircuit(nspins)
        for l in range(num_layers):
            for i in range(nspins):
                ##############
                q=np.random.randint(nspins)
                g=np.random.randint(1, 4)
                p=np.random.uniform(-1,1)
                if g==1:
                    circ.rx(p,q)
                if g==2:
                    circ.ry(p,q)
                if g==2:
                    circ.rz(p,q)
                ##############
                q=np.random.randint(nspins-1)
                circ.cnot(q, q+1)
        return circ            
    
    ################################################################
    ################################################################
    def getRandomQ(nspins):
        Q = np.random.randint(0,2, size=nspins)
        return Q
    
    ################################################################
    ################################################################
    def getCircUQU(circ, Q):
        circ_uhu=QuantumCircuit.copy(circ)
        for (q,o) in enumerate(Q):
            if o==0:
                pass
            if o==1:
                circ_uhu.x(q)
            if o==2:
                circ_uhu.y(q)
            if o==3:
                circ_uhu.z(q)     
                
        
        circ_uhu=circ_uhu.compose(QuantumCircuit.copy(circ).inverse())
        return circ_uhu
    
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
    shots = 1000
    eta = 100
    significant_figures = 3#np.log10(np.sqrt(shots)).astype(int)
    
    circ = getRandomU(nspins, num_layers)
    Q = getRandomQ(nspins)
    circ_uhu = getCircUQU(circ, Q)
    
        
    [counts, j_indxs], df_count = getIndexsFromExecute(circ_uhu, shots, backend = 'qasm_simulator')
    
    print(df_count)