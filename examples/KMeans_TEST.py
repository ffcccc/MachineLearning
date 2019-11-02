"""
@ Filename:       KMeans_TEST.py
@ Author:         Danc1elion
@ Create Date:    2019-05-16   
@ Update Date:    2019-05-28
@ Description:    Implement KMeans_TEST
"""
import time,sys,os
# LIB is the parent directory of the directory where program resides.
LIB = os.path.join(os.path.dirname(__file__), '..')
DAT = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'dataset6')
sys.path.insert(0, LIB)
import matplotlib.pyplot as plt
from Cluster import KMeans as kmeans
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

trainData = np.array(pd.read_table(os.path.join(DAT,'train.txt'), header=None, encoding='gb2312', delim_whitespace=True))
trainData = np.array(trainData)

time_start1 = time.time()
clf1 = kmeans(k=4, cluster_type="KMeans")
pred1 = clf1.train(trainData)
time_end1 = time.time()
print("Runtime of KMeans:", time_end1-time_start1)

time_start2 = time.time()
clf2 = kmeans(k=4, cluster_type="biKMeans")
pred = clf2.train(trainData)
time_end2 = time.time()
print("Runtime of biKMeans:", time_end2-time_start2)

time_start3 = time.time()
clf3 = kmeans(k=4, cluster_type="KMeans++")
pred3 = clf3.train(trainData)
time_end3 = time.time()
print("Runtime of KMeans++:", time_end3-time_start3)




