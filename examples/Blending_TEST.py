"""
@ Filename:       Blending_TEST.py
@ Author:         Danc1elion
@ Create Date:    2019-05-04
@ Update Date:    2019-05-04
@ Description:    Test Blending
"""
import time,sys,os
# LIB is the parent directory of the directory where program resides.
LIB = os.path.join(os.path.dirname(__file__), '..')
DAT = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'dataset3')
sys.path.insert(0, LIB)
from Blending import *
from Perceptron import *
from LogisticRegression import *
import numpy as np
import pandas as pd

trainData = np.array(pd.read_table(os.path.join(DAT,'train.txt'), header=None, encoding='gb2312', delim_whitespace=True))
testData = np.array(pd.read_table(os.path.join(DAT,'test.txt'), header=None, encoding='gb2312', delim_whitespace=True))
trainLabel = trainData[:, -1]
trainData = np.delete(trainData, -1, axis=1)
testLabel = testData[:, -1]
testData = np.delete(testData, -1, axis=1)

clfs = [PerceptronClassifier(), PerceptronClassifier(), LogisticRegressionClassifier(), LogisticRegressionClassifier()]

time_start1 = time.time()
clf1 = BlendingClassifier(classifier_set=clfs)
clf1.train(trainData, trainLabel)
clf1.predict(testData)
score1 = clf1.accuracy(testLabel)
time_end1 = time.time()
print("Accuracy of self-Blending: %f" % score1)
print("Runtime of self-Blending:", time_end1-time_start1)

time_start2 = time.time()
clf2 = LogisticRegressionClassifier()
clf2.train(trainData, trainLabel)
clf2.predict(testData)
score2 = clf2.accuracy(testLabel)
time_end2 = time.time()
print("Accuracy of self-Logistic: %f" % score2)
print("Runtime of self-Logistic:", time_end2-time_start2)

time_start3 = time.time()
clf3 = PerceptronClassifier()
clf3.train(trainData, trainLabel)
clf3.predict(testData)
score3 = clf3.accuracy(testLabel)
time_end3 = time.time()
print("Accuracy of self-Perceptron: %f" % score3)
print("Runtime of self-Perceptron:", time_end3-time_start3)


