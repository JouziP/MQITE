#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 17:53:16 2022

@author: pejmanjouzdani
"""
import logging
from time import time 




class TestCirc:
    
    def __init__(self, func, FLAG=False):
        self.func = func
        self.func.name = func.__name__
        self.FLAG=FLAG
        
    def __call__(self, *args, **kwargs):
        
        #### If the flag is ON
        if self.FLAG==True:
            try:
                results = self.test1(self.func , *args, **kwargs)
            except:
                raise
            
            # if successful do other tests...
            print(results)
            return results
            
        #### If the flag is OFF: just run the function
        else:
            return self.func(*args, **kwargs)
            
    
    @staticmethod
    def test1( func, *args, **kwargs):
        pass
        # time 
        start = time()
        
        
        results = func(*args, **kwargs)
        
        ### cmpare with hat is expected using *args etc. 
        end_time = time()
        
        logging.info('\n %s function takes %s second to run'%(func.name, end_time - start) )
        print('\n %s function takes %s second to run'%(func.name, end_time - start))
        
        return results
    
    def test2(self, *args, **kwargs):
        NotImplemented
        
        
        
        
    


if __name__=='__main__':
    
    from qiskit import QuantumCircuit

    from BasicFunctions.functions import getQCirc, getUQUCirc
    
    tested_func= TestCirc(getQCirc, FLAG=True)
    
    number_qubits= 3
    
    Q = [1,2,3]
    
    circ = QuantumCircuit(number_qubits)
    
    circ_Q  = tested_func(circ, Q)
    
    
    
    
    
    