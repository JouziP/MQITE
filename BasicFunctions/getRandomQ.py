#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 12:17:13 2022

@author: pejmanjouzdani
"""



import numpy as np

################################################################
################################################################
def getRandomQ(nspins):
    Q = np.random.randint(0,2, size=nspins)
    return Q