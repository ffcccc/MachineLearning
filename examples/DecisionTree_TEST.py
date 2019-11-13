import time,sys,os
# LIB is the parent directory of the directory where program resides.
LIB = os.path.join(os.path.dirname(__file__), '..')
DAT = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'dataset1')
sys.path.insert(0, LIB)
from sklearn import tree
from DecisionTree import *
import numpy as np
import pandas as pd

trainData = pd.read_table(os.path.join(DAT,'train.txt'), header=None, encoding='gb2312', delim_whitespace=True)
testData = pd.read_table(os.path.join(DAT,'test.txt'), header=None, encoding='gb2312', delim_whitespace=True)
trainLabel = np.array(trainData.pop(3))
trainData = np.array(trainData)
testLabel = np.array(testData.pop(3))
testData = np.array(testData)

time_start1 = time.time()
clf1 = DecisionTreeClassifier()
clf1.train(trainData, trainLabel)
clf1.predict(testData)
score1 = clf1.accuracy(testLabel)
time_end1 = time.time()
print("Accuracy of self-DecisionTree: %f" % score1)
print("Runtime of self-DecisionTree:", time_end1-time_start1)

time_start = time.time()
clf = tree.DecisionTreeClassifier()
clf.fit(trainData, trainLabel)
clf.predict(testData)
score = clf.score(testData, testLabel, sample_weight=None)
time_end = time.time()
print("Accuracy of sklearn-DecisionTree: %f" % score)
print("Runtime of sklearn-DecisionTree:", time_end-time_start)
