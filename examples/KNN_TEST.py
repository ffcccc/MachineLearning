import numpy as np
import pandas as pd
import time
import sys,os
# LIB is the parent directory of the directory where program resides.
LIB = os.path.join(os.path.dirname(__file__), '..')
DAT = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'dataset1')
sys.path.insert(0, LIB)
from sklearn.neighbors import KNeighborsClassifier
from KNN import *

trainData = pd.read_table(os.path.join(DAT,'train.txt'),header=None,encoding='gb2312',delim_whitespace=True)
testData = pd.read_table(os.path.join(DAT,'test.txt'),header=None,encoding='gb2312',delim_whitespace=True)
trainLabel = np.array(trainData.pop(3))
trainData = np.array(trainData)
testLabel = np.array(testData.pop(3))
testData = np.array(testData)

time_start1 = time.time()
clf1 = KNNClassifier(k=6)
clf1.train(trainData, trainLabel)
clf1.predict(testData)
# score1 = clf1.showDetectionResult(testData, testLabel)
score1 = clf1.accuracy(testLabel)
time_end1 = time.time()
print("Accuracy of self-KNN: %f" % score1)
print("Runtime of self-KNN:", time_end1-time_start1)

time_start = time.time()
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(trainData, trainLabel)
knn.predict(testData)
score = knn.score(testData, testLabel, sample_weight=None)
time_end = time.time()
print("Accuracy of sklearn-KNN: %f" % score)
print("Runtime of sklearn-KNN:", time_end-time_start)
