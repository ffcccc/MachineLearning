import numpy as np


# from cpp import distance as dis
# x = np.array([1.1, 2.2, 3.3, 4.4])
# x2 = np.array([1.1, 2.7, 3.3, 1.4])
# a = dis.distEuclidean(x, x2)

'''
Function:  Normalization
Description: Normalize input data. For vector x, the normalization process is given by
            normalization(x) = (x - min(x))/(max(x) - min(x))
Input:  data        dataType: ndarray   description: input data
Output: normdata    dataType: ndarray   description: output data after normalization
'''

def Normalization(data):
    # get the max and min value of each column
    minValue = data.min(axis=0)
    maxValue = data.max(axis=0)
    diff = maxValue - minValue
    # normalization
    mindata = np.tile(minValue, (data.shape[0], 1))
    normdata = (data - mindata) / np.tile(diff, (data.shape[0], 1))
    return normdata

'''
Function:  Standardization
Description: Standardize input data. For vector x, the normalization process is given by
             Standardization(x) = x - mean(x)/std(x)
Input:  data            dataType: ndarray   description: input data
Output: standarddata    dataType: ndarray   description: output data after standardization
'''

def Standardization(data):
    # get the mean and the variance of each column
    meanValue = data.mean(axis=0)
    varValue = data.std(axis=0)
    standarddata = (data - np.tile(meanValue, (data.shape[0], 1))) / np.tile(varValue, (data.shape[0], 1))
    return standarddata
<<<<<<< Updated upstream
=======

'''
Function:  calcuateDistance
Description: calcuate the distance between input vector and train data
Input:  x1      dataType: ndarray   description: input matrix: vector of samples (e.g. traindata)
        x2      dataType: ndarray   description: input vector to be calculated vs all the samples
Output: d       dataType: float     description: distance between input vectors
'''
def calculateDistance(distance_type, x1, x2):
    if distance_type == "Euclidean":
        #d = np.sqrt(np.sum(np.power(x1 - x2, 2)))
        d = np.sqrt(np.sum(np.power(x1 - x2, 2), axis=length(x1.shape))
    elif distance_type == "Cosine":
        d = np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))
    elif distance_type == "Manhattan":
        d = np.sum(x1 - x2)
    else:
        print("Error Type!")
        sys.exit()
    return d

'''
Function:  calcuateDistance
Description: calcuate the distance between input vector and train data
Input:  input       dataType: ndarray   description: input vector
        traind_ata  dataType: ndarray   description: data for training
        train_label dataType: ndarray   description: labels of train data
        k           dataType: int       description: select the first k distances
Output: prob        dataType: float     description: max probability of prediction 
        label       dataType: int       description: prediction label of input vector
'''
# def calculateDistance(input, train_data, train_label, k):
#     train_num = train_data.shape[0]
#     # calcuate the distances
#     distances = np.tile(input, (train_num, 1)) - train_data
#     distances = distances**2
#     distances = distances.sum(axis=1)
#     distances = distances**0.5

#     # get the labels of the first k distances
#     disIndex = distances.argsort()
#     labelCount = {}
#     for i in range(k):
#         label = train_label[disIndex[i]]
#         labelCount[label] = labelCount.get(label, 0) + 1

#     prediction = sorted(labelCount.items(), key=op.itemgetter(1), reverse=True)
#     label = prediction[0][0]
#     prob = prediction[0][1]/k
#     return label, prob

'''
Function:  calculateAccuracy
Description: show detection result
Input:  test_data  dataType: ndarray   description: data for test
        test_label dataType: ndarray   description: labels of test data
Output: accuracy   dataType: float     description: detection accuarcy
'''
def calculateAccuracy(test_label, prediction):
    test_label = np.expand_dims(test_label, axis=1)
    # prediction = self.prediction
    accuracy = sum(prediction == test_label)/len(test_label)
    return accuracy
>>>>>>> Stashed changes
