# Step-by-Step Guide to Implement Machine Learning VI - AdaBoost

![img](https://www.codeproject.com/script/Membership/ProfileImages/{6098a88e-6a47-4a71-915c-4577a7b84ee9}.jpg)

[danc1elion](https://www.codeproject.com/script/Membership/View.aspx?mid=14354398)

|      | Rate this: 			 				 			                      				   	 	         ![img](https://codeproject.freetls.fastly.net/script/Ratings/Images/stars-fill-lg.png)![img](https://codeproject.freetls.fastly.net/script/Ratings/Images/stars-empty-lg.png) 		  	 	5.00  (1 vote) |
| ---- | ------------------------------------------------------------ |
|      |                                                              |

​                                        20 May 2019[CPOL](http://www.codeproject.com/info/cpol10.aspx)                                    

Easy to implement machine learning



This article is an entry in our [Machine Learning and Artificial Intelligence Challenge](https://www.codeproject.com/Competitions/1024/The-Machine-Learning-and-Artificial-Intelligence-C.aspx). Articles in this sub-section are not required to be full articles so care should be taken when voting.

## Introduction

AdaBoost is an approach of **Boosting**, which is based on the principle that combining multiclassifiers can get a more accurate result in a complex environment.

## AdaBoost Model

The AdaBoost model consists of weak classifiers, weight update and classify.

### Weak Classifiers

AdaBoost combines weak classifiers with certain strategies to get a  strong classifier, as shown below. At each iteration, the weights of  samples which are wrongly classified will increase to catch the  classifier "attention". For example, in Fig. (a), the dotted line is the classifier-plane and there are two blue samples and one red sample  which are wrong classified. Then, in Fig. (b), the weights of two blue  samples and one red sample are increased. After adjusting the weights at each iteration, we can combine all the weak classifiers to get the  final strong classifier.

![Image 1](https://www.codeproject.com/KB/AI/4114375/cdac8dae-bfa9-42c7-88b5-99cfbced7fec.Png)

### Weight Update

There are two types of weight to update at each iteration, namely, the weight of each sample ![w_{i}](https://www.zhihu.com/equation?tex=w_%7Bi%7D) and the weight of each weak classifiers ![\alpha_{m}](https://www.zhihu.com/equation?tex=%5Calpha_%7Bm%7D). At the beginning, there are initialized as follows:

![w_{i}=\frac{1}{N}](https://www.zhihu.com/equation?tex=w_%7Bi%7D%3D%5Cfrac%7B1%7D%7BN%7D)

![\alpha_{m}=\frac{1}{M}](https://www.zhihu.com/equation?tex=%5Calpha_%7Bm%7D%3D%5Cfrac%7B1%7D%7BM%7D)

where `N`, *`M`* are the number of samples and the number of weak classifiers respectively.

AdaBoost trains a weak classifier at each iteration denoted as ![G_{m}\left(x\right)](https://www.zhihu.com/equation?tex=G_%7Bm%7D%5Cleft%28x%5Cright%29) whose training error is calculated as:

![e_{m}=\sum_{i=1}^{N}w_{mi}I\left(G_{m}(x_{i})\ne y_{i}\right)](https://www.zhihu.com/equation?tex=e_%7Bm%7D%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7Dw_%7Bmi%7DI%5Cleft%28G_%7Bm%7D%28x_%7Bi%7D%29%5Cne+y_%7Bi%7D%5Cright%29)

Then, update the weight of weak classifier by:

![\alpha_{m}=\frac{1}{2}\ln\frac{1-e_{m}}{e_{m}}](https://www.zhihu.com/equation?tex=%5Calpha_%7Bm%7D%3D%5Cfrac%7B1%7D%7B2%7D%5Cln%5Cfrac%7B1-e_%7Bm%7D%7D%7Be_%7Bm%7D%7D)

Update the weights of samples by:

![w_{mi}=\frac{w_{mi}\cdot \exp\left(-\alpha_{m}y_{i}G_{m}\left(x_{i}\right)\right)}{Z_{m}}](https://www.zhihu.com/equation?tex=w_%7Bmi%7D%3D%5Cfrac%7Bw_%7Bmi%7D%5Ccdot+%5Cexp%5Cleft%28-%5Calpha_%7Bm%7Dy_%7Bi%7DG_%7Bm%7D%5Cleft%28x_%7Bi%7D%5Cright%29%5Cright%29%7D%7BZ_%7Bm%7D%7D)

where:

![Z_{m}=\sum_{i=1}^{N}w_{mi}\exp\left(-\alpha_{m}y_{i}G_{m}\left(x_{i}\right)\right)](https://www.zhihu.com/equation?tex=Z_%7Bm%7D%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7Dw_%7Bmi%7D%5Cexp%5Cleft%28-%5Calpha_%7Bm%7Dy_%7Bi%7DG_%7Bm%7D%5Cleft%28x_%7Bi%7D%5Cright%29%5Cright%29)

From the above equations, we can conclude that:

1. The training error is the sum of weights of the wrong classified samples.

2. When *em* is less than 0.5, *am* is greater than 0, which means the lower training error the weak  classifiers has, the more important role that weak classifier plays in  the final classifier.

3. The weight update can be written as: 

   ![w_{mi}=\left\{ \begin{aligned} \frac{w_{mi}}{Z_{m}}e^{-\alpha_{m}},G_{m}\left(x_{i}\right)=y_{i}\\ \frac{w_{mi}}{Z_{m}}e^{\alpha_{m}},G_{m}\left(x_{i}\right)\ne y_{i} \end{aligned} \right. ](https://www.zhihu.com/equation?tex=w_%7Bmi%7D%3D%5Cleft%5C%7B+%5Cbegin%7Baligned%7D+%5Cfrac%7Bw_%7Bmi%7D%7D%7BZ_%7Bm%7D%7De%5E%7B-%5Calpha_%7Bm%7D%7D%2CG_%7Bm%7D%5Cleft%28x_%7Bi%7D%5Cright%29%3Dy_%7Bi%7D%5C%5C+%5Cfrac%7Bw_%7Bmi%7D%7D%7BZ_%7Bm%7D%7De%5E%7B%5Calpha_%7Bm%7D%7D%2CG_%7Bm%7D%5Cleft%28x_%7Bi%7D%5Cright%29%5Cne+y_%7Bi%7D+%5Cend%7Baligned%7D+%5Cright.+)

   which means that the weights of right classified samples decrease while the weights of wrong classified samples increase.

The code of training process of AdaBoost is shown below:

Hide   Shrink ![img](https://www.codeproject.com/images/arrow-up-16.png)   Copy Code

```
def train(self, train_data, train_label):
        if self.norm_type == "Standardization":
            train_data = preProcess.Standardization(train_data)
        else:
            train_data = preProcess.Normalization(train_data)

        train_label = np.expand_dims(train_label, axis=1)
        sample_num = len(train_data)

        weak_classifier = []

        # initialize weights
        w = np.ones([sample_num, 1])
        w = w/sample_num

        # predictions
        agg_predicts = np.zeros([sample_num, 1]) # aggregate value of prediction

        # start train
        for i in range(self.iterations):
            base_clf, error, base_prediction = self.baseClassifier(train_data, train_label, w)
            alpha = self.updateAlpha(error)
            weak_classifier.append((alpha, base_clf))

            # update parameters in page of 139 Eq.(8.4)
            expon = np.multiply(-1 * alpha * train_label, base_prediction)
            w = np.multiply(w, np.exp(expon))
            w = w/w.sum()

            # calculate the total error rate
            agg_predicts += alpha*base_prediction
            error_rate = np.multiply(np.sign(agg_predicts) != train_label, 
                         np.ones([sample_num, 1]))
            error_rate = error_rate.sum()/sample_num

            if error_rate == 0:
                break
            self.classifier_set = weak_classifier
        return weak_classifier
```

### Classify

Combine all the weak classifiers to get a strong classifier. The  classify rule is the weighted sum of each weak classifier result, which  is given by:

![G\left(x\right)=sign\left(\sum_{m=1}^{M}\alpha_{m}G_{m}\left(x\right)\right)](https://www.zhihu.com/equation?tex=G%5Cleft%28x%5Cright%29%3Dsign%5Cleft%28%5Csum_%7Bm%3D1%7D%5E%7BM%7D%5Calpha_%7Bm%7DG_%7Bm%7D%5Cleft%28x%5Cright%29%5Cright%29)

## Conclusion and Analysis

AdaBoost can be regarded as additive model with exponent loss  function using forward forward step algorithm. In AdaBoost, the type of  weak classifiers can be different or the same. In this article, we use 5 SVM classifiers as the weak classifiers, and the detection performance  is shown below:

![Image 13](https://www.codeproject.com/KB/AI/4114375/38b557d2-0200-4ac1-9030-a63d8d4bc774.Png)

It can be that the accuracy increases about 5% and the runtime is about 5 times of the single SVM.

The related code and dataset in this article can be found in [MachineLearning](https://github.com/DandelionLau/MachineLearning).