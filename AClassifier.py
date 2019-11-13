"""
@ Filename:       AClassifier.py
@ Author:         ffcccc
@ Create Date:    2019-11-07   
@ Update Date:    2019-11-07 
@ Description:    Classification base class
"""

import numpy as np
import pickle
import preProcess

class aClassifier:
    def __init__(self):
        self.prediction = None
        self.probability = None

    '''
    Function:  accuracy
    Description: show detection result
    Input:  test_data  dataType: ndarray   description: data for test
            test_label dataType: ndarray   description: labels of test data
    Output: accuracy   dataType: float     description: detection accuarcy
    '''
    def accuracy(self, test_label):
        # test_label = np.expand_dims(test_label, axis=1)
        # prediction = self.prediction
        # accuarcy = sum(prediction == test_label)/len(test_label)
        local_accuracy = preProcess.calculateAccuracy(test_label, self.prediction)
        # calculateAccuracy(test_label, self.prediction):
        return local_accuracy