"""
@ Filename:       Apriori_TEST.py
@ Author:         Danc1elion
@ Create Date:    2019-05-28   
@ Update Date:    2019-05-31 
@ Description:    Implement Apriori_TEST
"""
import time,sys,os
LIB = os.path.join(os.path.dirname(__file__), '..')
DAT = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'dataset3')
sys.path.insert(0, LIB)
from AssociationAnalysis import Apriori
import numpy as np
import pandas as pd

trainData = [['bread', 'milk', 'vegetable', 'fruit', 'eggs'],
           ['noodle', 'beef', 'pork', 'water', 'socks', 'gloves', 'shoes', 'rice'],
           ['socks', 'gloves'],
           ['bread', 'milk', 'shoes', 'socks', 'eggs'],
           ['socks', 'shoes', 'sweater', 'cap', 'milk', 'vegetable', 'gloves'],
           ['eggs', 'bread', 'milk', 'fish', 'crab', 'shrimp', 'rice']]

time_start1 = time.time()
clf1 = Apriori()
pred1 = clf1.train(trainData)
time_end1 = time.time()
print("Runtime of Apriori:", time_end1-time_start1)
