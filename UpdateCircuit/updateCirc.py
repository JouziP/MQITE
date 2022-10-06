#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 19:06:39 2022

@author: pej
"""


import numpy as np
import pandas as pd

from qiskit import QuantumCircuit





from BasicFunctions.functions import timing, getBinary, getGateStats
from MultiQubitGate.functions import multiqubit
#############################
# @timing
def updateCirc(list_j, list_y_j, circ,
                   ):
    
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


if __name__=='__main__':
    '''
    Try to catch scenarios where there are significant errors
    
    '''
    
    pass


      
    import numpy as np
    import pandas as pd
    
    
    
    from qiskit import QuantumCircuit
    
    
    
    from BasicFunctions.functions import getBinary, getState
        
    nspins = 2
    n_h = nspins
    ngates = nspins
    num_layers = 1
    num_itr =3
    W_max =1
    W_min = -1
    Hz_max=1
    Hz_min=-1
    k = 2
    delta0 = 0.1
    T = num_itr * delta0
    machine_precision = 10        
    chi  = 1
    shots = 10000
    significant_digits = 6#np.log10(np.sqrt(shots)).astype(int)
    delta = delta0
    seed = 2212
    ######
    np.random.seed(seed)
    
    
    ################################################################
    ################################################################
    ##################### FOR TEST           #######################
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
    
    def getRandomQ(nspins):
        Q = np.random.randint(0,2, size=nspins)
        return Q
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
    
    ################################# TEST
    ######## random circ
    circ = getRandomU(nspins, num_layers)
    # print(circ)
    #######
    Q = getRandomQ(nspins)    
    #######
    
    from UpdateCircuit.findComponents import findComponnets
    [list_j, list_y_j] = findComponnets(circ,
                   Q,                   
                   delta, 
                   significant_digits,                    
                   shots, 
                   nspins, 
                   norm=1)
    
    
    ########################
    print(circ.draw())
    circ_new = updateCirc(list_j, list_y_j, circ,
                   )
    
    print('---------')
    print(circ_new.draw())

















