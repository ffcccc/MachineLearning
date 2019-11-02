import numpy as np
import pandas as pd
import time,sys,os
# LIB is the parent directory of the directory where program resides.
LIB = os.path.join(os.path.dirname(__file__), '..')
DAT = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'dataset2')
sys.path.insert(0, LIB)
from AdaBoost import *
from SVM import *

trainData = np.array(pd.read_table(os.path.join(DAT,'train.txt'), header=None, encoding='gb2312', delim_whitespace=True))
testData = np.array(pd.read_table(os.path.join(DAT,'test.txt'), header=None, encoding='gb2312', delim_whitespace=True))
trainLabel = trainData[:, -1]
trainData = np.delete(trainData, -1, axis=1)
testLabel = testData[:, -1]
testData = np.delete(testData, -1, axis=1)

time_start1 = time.time()
clf1 = SVMClassifier()
clf1.train(trainData, trainLabel)
clf1.predict(testData)
score1 = clf1.accuarcy(testLabel)
time_end1 = time.time()
print("Accuracy of SVM: %f" % score1)
print("Runtime of SVM:", time_end1-time_start1)

time_start1 = time.time()
clf1 = Adaboost()
clf1.train(trainData, trainLabel)
clf1.predict(testData)
score1 = clf1.accuarcy(testLabel)
time_end1 = time.time()
print("Accuracy of Adaboost: %f" % score1)
print("Runtime of Adaboost:", time_end1-time_start1)
