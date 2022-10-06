#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:35:49 2021

@author: jouzdanip
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:03:07 2021

@author: jouzdanip
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 22:04:45 2021

@author: jouzdanip
"""
import time
import numpy as np
from numpy import linalg as lg

from qiskit.extensions import IGate,XGate,YGate,ZGate


I = IGate().to_matrix()
X = XGate().to_matrix()
Y = YGate().to_matrix()
Z = ZGate().to_matrix()


        
def getState(n):
    """ Get the qubit state """
    s=[np.matrix([[1],[0]]),np.matrix([[0],[1]])]
    for (i,q) in enumerate(n):
        if i==0:
            state = s[q]
        else:
            state = np.kron(state, s[q])
    return state.T.tolist()[0]

def constructExactHamiltonian(hamiltonian,weights,
                             ):
    """ Construct the exact hamiltonian from Pauli strings and coefficients """
   
    operatorset=[I,X,Y,Z]
        
    n = 2**len(hamiltonian[0])
    exact_hamiltonian= np.zeros([n,n], dtype=complex)

    for (el, h) in enumerate(hamiltonian):
            for q in range(len(h)):
    
                if q==0 :
                    O = operatorset[h[q]]    
                else:
                    O = np.kron(O, operatorset[h[q]])
            exact_hamiltonian += O * weights[el]
            print(el, end='-')
    return exact_hamiltonian
