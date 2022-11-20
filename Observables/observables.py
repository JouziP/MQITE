#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 11:48:30 2022

@author: pejmanjouzdani
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from BasicFunctions.getState import getState

class Observables:
    def __init__(self,observable_matrix, machine_precision , exact_value=None):
        self.observable_matrix = observable_matrix
        self.expectations_t = []
        self.machine_precision = machine_precision
        self.exact_value = exact_value
    
    def getExpectation_matrixVersion(self, 
                                     qiskit_circuit, t, k):
        
        expectation = self._getExpectation_matrixVersion(self.observable_matrix,
                                                        qiskit_circuit,
                                                        self.machine_precision)
        self.current_expectation = expectation
        self.expectations_t.append(
            [
                t,
                k,
                expectation
             ]
            )
    
    @staticmethod
    def _getExpectation_matrixVersion(observable_matrix, qiskit_circuit,
                                      machine_precision):
    
        state_vec = getState(qiskit_circuit, machine_precision)
        
        expectation = ((state_vec.T.conjugate().dot(observable_matrix)).dot(state_vec))[0,0].real
        
        return expectation
    
    def display(self, ):
        print('=============   From Observables ')
        print('------ current_expectation')
        print(self.current_expectation)
        print()
        if self.exact_value!=None:
            print('------ exact expectation')
            print(self.exact_value)
            print()
       
        
        fig, ax = plt.subplots(1,1 , figsize=[10, 8])
        expectations_t = pd.DataFrame(self.expectations_t)
        xs = [i for i in range(len(self.expectations_t))]
        ax.plot(xs, expectations_t[2].values, label='Expectaion')
        ax.set_xlabel('(t,k) step ')
        ax.set_ylabel('expectaion ')
        ax.legend()
        if self.exact_value!=None:
            ax.hlines(y=self.exact_value, xmin=xs[0], xmax=xs[-1])
        
        plt.show()
        plt.close()
    
        
    
    