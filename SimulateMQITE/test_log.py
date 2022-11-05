#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:20:18 2022

@author: pejmanjouzdani
"""

import os
import logging
filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ram.log')
logging.basicConfig(filename=filename, level=logging.DEBUG, force=True)
logging.debug('This message should go to the log file')