# Step-by-Step Guide To Implement Machine Learning I - KNN

![img](https://www.codeproject.com/script/Membership/ProfileImages/{6098a88e-6a47-4a71-915c-4577a7b84ee9}.jpg)

[danc1elion](https://www.codeproject.com/script/Membership/View.aspx?mid=14354398)

|      | Rate this: 			 				 			                      				   	 	         ![img](https://codeproject.freetls.fastly.net/script/Ratings/Images/stars-fill-lg.png)![img](https://codeproject.freetls.fastly.net/script/Ratings/Images/stars-empty-lg.png) 		  	 	2.25  (5 votes) |
| ---- | ------------------------------------------------------------ |
|      |                                                              |

​                                        20 May 2019[CPOL](http://www.codeproject.com/info/cpol10.aspx)                                    

Easy to implement machine learning



This article is an entry in our [Machine Learning and Artificial Intelligence Challenge](https://www.codeproject.com/Competitions/1024/The-Machine-Learning-and-Artificial-Intelligence-C.aspx). Articles in this sub-section are not required to be full articles so care should be taken when voting.

## Introduction

K-nearest neighbors(**KNN**) is a simple machine  learning algorithm, the principle of which is to calculate the distance  between the test object and the training set. Then, the object in  training set is selected by the distances to add to a K-NN set until the K-NN set includes a predefined number of nearest neighbors, which can  be expressed as:

![y = argmax\sum_{x_{j}\in N_{k}}^{k}{I(y_{i}=c_{i})}](https://www.zhihu.com/equation?tex=y+%3D+argmax%5Csum_%7Bx_%7Bj%7D%5Cin+N_%7Bk%7D%7D%5E%7Bk%7D%7BI%28y_%7Bi%7D%3Dc_%7Bi%7D%29%7D)

where ![N_{k}](https://www.zhihu.com/equation?tex=N_%7Bk%7D) is the KNN set, ![I](https://www.zhihu.com/equation?tex=I) is given by:

![I(x)= \begin{cases} 0,& \text{if x is true}\\ 1,& \text{else} \end{cases}](https://www.zhihu.com/equation?tex=I%28x%29%3D+%5Cbegin%7Bcases%7D+0%2C%26+%5Ctext%7Bif+x+is+true%7D%5C%5C+1%2C%26+%5Ctext%7Belse%7D+%5Cend%7Bcases%7D)

## KNN Model

KNN model consists of distance calculation, select K and classify.

### Distance Calculation

Distance is used to measure the similarity between two objects in the feature space. There are many methods to calculate distance between two objects, such as Euclidean distance, Cosine distance, Edit distance,  Manhattan distance, etc. The simplest method is **Euclidean distance**, which is calculated as:

![L_{2}(x_{i},x_{j})=\left( \sum_{l=1}^{n}{|x_{i}^{(l)}-x_{j}^{(l)}|^{2}}\right)^\frac{1}{2}](https://www.zhihu.com/equation?tex=L_%7B2%7D%28x_%7Bi%7D%2Cx_%7Bj%7D%29%3D%5Cleft%28+%5Csum_%7Bl%3D1%7D%5E%7Bn%7D%7B%7Cx_%7Bi%7D%5E%7B%28l%29%7D-x_%7Bj%7D%5E%7B%28l%29%7D%7C%5E%7B2%7D%7D%5Cright%29%5E%5Cfrac%7B1%7D%7B2%7D)

where `*n*` is the feature dimension.

Because the different scales of feature value, the bigger value has  more effect on the distance. Thus, all the features need to be  normalized. Specifically, there are two normalization methods. One is  min-max normalization, which is given by:

![x'=\frac{x-min(x)}{max(x)-min(x)}](https://www.zhihu.com/equation?tex=x%27%3D%5Cfrac%7Bx-min%28x%29%7D%7Bmax%28x%29-min%28x%29%7D)

Hide   Copy Code

```
def Normalization(self, data):
    # get the max and min value of each column
    minValue = data.min(axis=0)
    maxValue = data.max(axis=0)
    diff = maxValue - minValue
    # normalization
    mindata = np.tile(minValue, (data.shape[0], 1))
    normdata = (data - mindata)/np.tile(diff, (data.shape[0], 1))
    return normdata
```

The other is z-score normalization, which is given by:

![x'=\frac{x-\mu}{\sigma}](https://www.zhihu.com/equation?tex=x%27%3D%5Cfrac%7Bx-%5Cmu%7D%7B%5Csigma%7D)

where ![\mu](https://www.zhihu.com/equation?tex=%5Cmu) is the mean of ![x](https://www.zhihu.com/equation?tex=x) and ![\sigma](https://www.zhihu.com/equation?tex=%5Csigma) is the standard deviation of ![x](https://www.zhihu.com/equation?tex=x).

Hide   Copy Code

```
def Standardization(self, data):
    # get the mean and the variance of each column
    meanValue = data.mean(axis=0)
    varValue = data.std(axis=0)
    standarddata = (data - np.tile(meanValue,
                   (data.shape[0], 1)))/np.tile(varValue, (data.shape[0], 1))
    return standarddata
```

After normalization, the code of Euclidean distance is shown below:

Hide   Copy Code

```
train_num = train_data.shape[0]
# calculate the distances
distances = np.tile(input, (train_num, 1)) - train_data
distances = distances**2
distances = distances.sum(axis=1)
distances = distances**0.5
```

### Select K

If we select a small K, the model is learned with a small neighbor which will lead to small "**approximate error**" while large "**estimate error**". In a word, small K will make the model complex and tend to overfit. On  the contrary, if we select a large K, the model is learned with a large  neighbor which will lead to large "**approximate error**" while small "**estimate error**". In a word, large K will make the model simple and tend to large computation.

### Classifiy

After determine K, we can utilize vote for classification, which  means the minority is subordinate to the majority, whose code is shown  below:

Hide   Copy Code

```
disIndex = distances.argsort()
labelCount = {}
for i in range(k):
    label = train_label[disIndex[i]]
    labelCount[label] = labelCount.get(label, 0) + 1
prediction = sorted(labelCount.items(), key=op.itemgetter(1), reverse=True)
label = prediction[0][0]
```

In the above code, we first use `argsorts()` to get the ordered index, then count each kind of label in the first K samples, finally `labelCount` is sorted to get the label with the most votes, which is the prediction for the test object. The whole prediction function is shown below:

Hide   Copy Code

```
def calcuateDistance(self, input_sample, train_data, train_label, k):
        train_num = train_data.shape[0]
        # calculate the distances
        distances = np.tile(input, (train_num, 1)) - train_data
        distances = distances**2
        distances = distances.sum(axis=1)
        distances = distances**0.5

        # get the labels of the first k distances
        disIndex = distances.argsort()
        labelCount = {}
        for i in range(k):
            label = train_label[disIndex[i]]
            labelCount[label] = labelCount.get(label, 0) + 1

        prediction = sorted(labelCount.items(), key=op.itemgetter(1), reverse=True)
        label = prediction[0][0]
        return label
```

## Conclusion and Analysis

KNN is implemented by linear traverse in this articles. However,  there exists more effective method for KNN, like kd tree. Moreover, it  is valid to apply cross-validation to get a more suitable K. Finally,  let's compare our KNN with the KNN in Sklearn and the detection  performance is displayed below.

![Image 12](https://www.codeproject.com/KB/AI/4044571/fba3aedc-45a4-4af8-a47c-7512079025a3.Png)

Form the figure, we can learn that the KNN in this article is better  than sklearn knn in terms of accuracy. The runtime is nearly the same.

The related code and dataset in this article can be found in [MachineLearning](https://github.com/DandelionLau/MachineLearning).