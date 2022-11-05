#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 15:03:00 2022

@author: pejmanjouzdani
"""
"""
Simple test using logging
"""
class A:
    def __init__(self, x):
        return self.__call__(x)
    
    def __call__(self, x):
        return x+1
    
if __name__=='__main__':
    x = 10
    print(A(x))

    