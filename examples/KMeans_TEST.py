"""
@ Filename:       KMeans_TEST.py
@ Author:         Ryuk
@ Create Date:    2019-05-16   
@ Update Date:    2019-05-28
@ Description:    Implement KMeans_TEST
"""
import matplotlib.pyplot as plt
from Cluster import KMeans as kmeans
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import time

trainData = pd.read_table('../dataset/dataset6/train.txt',header=None,encoding='gb2312', delim_whitespace=True)
trainData = np.array(trainData)

# sample_num = len(train_data)
# feat_num = trainData.shape[1]
# d = np.zeros([sample_num, feat_num])
# d2 = np.zeros([sample_num, feat_num])

# for i in range(len(centers)):
#     # calculate the distance between each sample and each cluster center
#     # d[:, i] = self.calculateDistance(train_data, centers[i])
#     bb = preProcess.calculateDistance(self.distance_type, train_data, centers[i])
#     d[:, i] = bb
#     # for j in range(sample_num):
#     #     aa = cdist.distEuclidean(train_data[j, :], centers[i])
#     #     d[j, i] = aa

time_start1 = time.time()
clf1 = kmeans(k=4, cluster_type="KMeans")
pred1 = clf1.train(trainData, display=False)
time_end1 = time.time()
print("Runtime of KMeans:", time_end1-time_start1)
clf1.plotResult(trainData)

time_start2 = time.time()
clf2 = kmeans(k=4, cluster_type="biKMeans")
pred = clf2.train(trainData, display=False)
time_end2 = time.time()
print("Runtime of biKMeans:", time_end2-time_start2)
clf2.plotResult(trainData)

time_start3 = time.time()
clf3 = kmeans(k=4, cluster_type="KMeans++")
pred3 = clf3.train(trainData, display=False)
time_end3 = time.time()
print("Runtime of KMeans++:", time_end3-time_start3)
clf3.plotResult(trainData)



