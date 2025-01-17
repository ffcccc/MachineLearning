"""
@ Filename:       Eclat_TEST.py
@ Author:         Danc1elion
@ Create Date:    2019-06-02   
@ Update Date:    2019-06-02 
@ Description:    Implement Eclat_TEST
"""

import time,sys,os
LIB = os.path.join(os.path.dirname(__file__), '..')
DAT = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'dataset3')
sys.path.insert(0, LIB)
from AssociationAnalysis import Eclat
import numpy as np
import pandas as pd

trainData = [['bread', 'milk', 'vegetable', 'fruit', 'eggs'],
            ['noodle', 'beef', 'pork', 'water', 'socks', 'gloves', 'shoes', 'rice'],
            ['socks', 'gloves'],
            ['bread', 'milk', 'shoes', 'socks', 'eggs'],
            ['socks', 'shoes', 'sweater', 'cap', 'milk', 'vegetable', 'gloves'],
            ['eggs', 'bread', 'milk', 'fish', 'crab', 'shrimp', 'rice']]

time_start1 = time.time()
clf1 = Eclat()
pred1 = clf1.train(trainData)
time_end1 = time.time()
print("Runtime of Eclat:", time_end1-time_start1)
