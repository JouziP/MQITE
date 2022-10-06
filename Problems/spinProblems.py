#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:11:38 2022

@author: pej
"""

      
from time import time    

import numpy as np
from numpy import linalg as lg
from scipy.sparse.linalg import cg
import pandas as pd
from matplotlib import pyplot as plt
from functools import wraps

from qiskit.extensions import IGate,XGate,YGate,ZGate
from qiskit import QuantumCircuit, execute, BasicAer
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import partial_trace, Statevector
from qiskit import Aer


from MultiQubitGate.functions import multiqubit

from Problems.getNuclearExample import getNuclearExample


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r --->  %2.4f sec' %(f.__name__,   te-ts) )
        return result
    return wrap

X_op = np.matrix([[0.+0.j, 1.+0.j],
                       [1.+0.j, 0.+0.j]])
Y_op = np.matrix([[ 0.+0.j, -0.-1.j],
               [ 0.+1.j,  0.+0.j]])
Z_op = np.matrix([[ 1.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j]])
I2 = np.matrix([[1.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j]])

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

def getBinary(j, nspins):
    string = np.array(list(np.binary_repr(j, nspins))).astype(int)[::-1]
    return string


def generateRandom_XX_Z(nspins, n_h):
    ### ZZ and X on random bonds
    hs = []
    for j in range(n_h):
        indxs = [i for i in range(nspins)]
        ### ZZ
        np.random.shuffle(indxs)
        [i1, i2] = [indxs[0], indxs[1] ] 
        h = [0 for i in range(nspins)]
        h[i1] = 1
        h[i2] = 1
        hs.append(h)
        

    indxs = [i for i in range(nspins)]        
    for i1 in indxs:    
        ### X
        
        
        h = [0 for i in range(nspins)]
        h[i1] = 3       
        hs.append(h)    
    return hs 




def generateRandom_YY_Z(nspins, n_h):
    ### ZZ and X on random bonds
    hs = []
    for j in range(n_h):
        indxs = [i for i in range(nspins)]
        ### ZZ
        np.random.shuffle(indxs)
        [i1, i2] = [indxs[0], indxs[1] ] 
        h = [0 for i in range(nspins)]
        h[i1] = 1
        h[i2] = 1
        hs.append(h)
        
    for j in range(n_h):
        indxs = [i for i in range(nspins)]        
        ### X
        np.random.shuffle(indxs)
        i1 = indxs[0]
        h = [0 for i in range(nspins)]
        h[i1] = 3        
        hs.append(h)    
    return hs 



def generateRandom_k_local_XZ(nspins, n_h , k):
    '''
        Y and X on random bonds
    '''
    hs = []
    for j in range(n_h):
        indxs = [i for i in range(nspins)]    
        np.random.shuffle(indxs)
        
        h = [0 for i in range(nspins)]
        
        os = [1,2]    
        for l in range(k):
            np.random.shuffle(os)
            h[indxs[l]]=os[0]
        
        hs.append(h)
        
    
    return hs 





def generateRandom_k_local_XY(nspins, n_h , k):
    ### ZZ and X on random bonds
    hs = []
    for j in range(n_h):
        indxs = [i for i in range(nspins)]    
        np.random.shuffle(indxs)
        
        h = [0 for i in range(nspins)]
        
        os = [1,2]    
        for l in range(k):
            np.random.shuffle(os)
            h[indxs[l]]=os[0]
        
        hs.append(h)
        
    
    return hs 




def generateRandom_k_local_XYZ(nspins, n_h , k):
    ### ZZ and X on random bonds
    hs = []
    for j in range(n_h):
        indxs = [i for i in range(nspins)]    
        np.random.shuffle(indxs)
        
        h = [0 for i in range(nspins)]
        
        os = [1,2, 3]    
        for l in range(k):
            np.random.shuffle(os)
            h[indxs[l]]=os[0]
        
        hs.append(h)
        
    
    return hs 


##################


class RandomIsing():
    def __init__(self, nspins, num_neighbors):
        self.nspins = nspins
        self.num_neighbors = num_neighbors
        self.edges_dic = {}
        self.edges_count_dic = {}
        self.edges = []
        for n in range(nspins):
            self.edges_dic[str(n)] = []
            self.edges_count_dic[str(n)] = 0
        self.defineEdges()
        
    def defineEdges(self):
        nspins = self.nspins
        num_neighbors = self.num_neighbors
        for n in range(nspins):
            neighbs = [k for k in range(n+1, nspins)]
            
            np.random.shuffle(neighbs)
            
             
            
            
            for k in neighbs:
                self.edges_dic[str(n)].append(k) 
                self.edges_dic[str(k)].append(n)
                
        
        
            
        for n in range(nspins):
            for k in self.edges_dic[str(n)]:
                if self.edges_count_dic[str(n)]<num_neighbors:
                    if self.edges_count_dic[str(k)]<num_neighbors:                    
                        self.edges.append([n,k])
                        self.edges_count_dic[str(n)] +=1
                        self.edges_count_dic[str(k)] +=1
        
        for n in range(nspins):
            
            while self.edges_count_dic[str(n)]<num_neighbors:
                neighbs = [k for k in range(nspins)]
                neighbs.pop(n)
                np.random.shuffle(neighbs)
                k = neighbs[0]
                self.edges.append([n,k])
                self.edges_count_dic[str(n)] +=1
                self.edges_count_dic[str(k)] +=1
    
    def getRandH_ZZ(self):
        nspins = self.nspins
        hs = []
        for edge in self.edges:
            h = [0 for i in range(nspins)]
            
            n = edge[0]
            k = edge[1]
            h[n] = 1
            h[k] = 1
            hs.append(h)
        
    
        return hs 
    
    
    def getRandH_XX_Z(self):
        nspins = self.nspins
        hs = []
        for edge in self.edges:
            h = [0 for i in range(nspins)]
            
            n = edge[0]
            k = edge[1]
            h[n] = 1
            h[k] = 1
            hs.append(h)
        ##### remove repetitions
        vs = []
        for j in range(len(hs)) :
            v = sum([s*2**i for (i,s) in  enumerate(hs[j]) ])
            vs.append(v)
        vs  = sorted(list(set(vs)))
        hs = []
        for v in vs:
            hs.append(getBinary(v, nspins).tolist())
            
        
        for n in range(nspins):            
            ### Z
            h = [0 for i in range(nspins)]
            h[n] = 3
            hs.append(h)
        
    
        return hs 
    





class NuclearExample():
    def __init__(self,filename):
        self.filename= filename
        self.files = [
            'ham-0p-pn-spherical.txt',
            'ham-0p-spherical.txt',
            'ham-0p32.txt',
            'ham-JW-full.txt',
            ]
        if filename in self.files:
            hs, ws = getNuclearExample(filename)
            self.hs = hs 
            self.ws = ws
            self.nspins= len(hs[0])
            self.nh= len(hs)
            self.__name__  = 'NuclearExample'+'_'+self.filename
            
            
            
        


class Ising():
    def __init__(self,nspins):        
        self.nspins= nspins
    
    def oneDTIM(self):
        nspins = self.nspins
        hs = []
        for q in range(nspins-1):
            h = [0 for i in range(nspins)]
            h[q] = 3
            h[q+1] = 3
            hs.append(h)
        for q in range(nspins):
            h = [0 for i in range(nspins)]
            h[q] = 1
            hs.append(h)
        return hs
            
                    
                    






        
    
if __name__=='__main__':
    seed = 12321
    # np.random.seed(seed)
                
    # r=RandomIsing(10, 4)    
    # hs = r.getRandH_XX_Z()
    
    # print(np.array(hs))
            
    
    # im = Ising(10)


























