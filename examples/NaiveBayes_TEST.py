"""
@ Filename:       NaiveBayes_TEST.py
@ Author:         Danc1elion
@ Create Date:    2019-05-16   
@ Update Date:    2019-05-16 
@ Description:    Implement NaiveBayes_TEST
"""
import time,sys,os
# LIB is the parent directory of the directory where program resides.
LIB = os.path.join(os.path.dirname(__file__), '..')
DAT = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'dataset1')
sys.path.insert(0, LIB)
from sklearn.naive_bayes import BernoulliNB
from NaiveBayes import *
import numpy as np
import pandas as pd

trainData = np.array(pd.read_table(os.path.join(DAT,'train.txt'), header=None, encoding='gb2312', delim_whitespace=True))
testData = np.array(pd.read_table(os.path.join(DAT,'test.txt'), header=None, encoding='gb2312', delim_whitespace=True))
trainLabel = np.array(trainData.pop(3))
trainData = np.array(trainData)
testLabel = np.array(testData.pop(3))
testData = np.array(testData)

time_start1 = time.time()
clf1 = BayesClassifier()
clf1.train(trainData, trainLabel)
clf1.predict(testData)
score1 = clf1.accuarcy(testLabel)
time_end1 = time.time()
print("Accuracy of self-Bayes: %f" % score1)
print("Runtime of self-Bayes:", time_end1-time_start1)

time_start = time.time()
clf = BernoulliNB()
clf.fit(trainData, trainLabel)
clf.predict(testData)
score = clf.score(testData, testLabel, sample_weight=None)
time_end = time.time()
print("Accuracy of sklearn-Bayes: %f" % score)
print("Runtime of sklearn-Bayes:", time_end-time_start)
