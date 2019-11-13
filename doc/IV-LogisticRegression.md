# Step-by-Step Guide to Implement Machine Learning IV - Logistic Regression

![img](https://www.codeproject.com/script/Membership/ProfileImages/{6098a88e-6a47-4a71-915c-4577a7b84ee9}.jpg)

[danc1elion](https://www.codeproject.com/script/Membership/View.aspx?mid=14354398)

|      | Rate this: 			 				 			                      				   	 	         ![img](https://codeproject.freetls.fastly.net/script/Ratings/Images/stars-fill-lg.png)![img](https://codeproject.freetls.fastly.net/script/Ratings/Images/stars-empty-lg.png) 		  	 	4.88  (6 votes) |
| ---- | ------------------------------------------------------------ |
|      |                                                              |

​                                        13 May 2019[CPOL](http://www.codeproject.com/info/cpol10.aspx)                                    

Easy to implement machine learning



This article is an entry in our [Machine Learning and Artificial Intelligence Challenge](https://www.codeproject.com/Competitions/1024/The-Machine-Learning-and-Artificial-Intelligence-C.aspx). Articles in this sub-section are not required to be full articles so care should be taken when voting.

## Introduction

Logisitic regression is a classical method in statistical learning, which calculates the conditional probability `P(Y|X)` and uses the label of the larger one as the prediction. Specifically, the binomial logistic regression model is:

![P\left(Y=1|x\right)=\frac{exp\left(w\cdot x+b\right)}{ 1 + exp\left(w\cdot x+b\right)}](https://www.zhihu.com/equation?tex=P%5Cleft%28Y%3D1%7Cx%5Cright%29%3D%5Cfrac%7Bexp%5Cleft%28w%5Ccdot+x%2Bb%5Cright%29%7D%7B+1+%2B+exp%5Cleft%28w%5Ccdot+x%2Bb%5Cright%29%7D)

![P\left(Y=0|x\right)=\frac{1}{ 1 + exp\left(w\cdot x+b\right)}](https://www.zhihu.com/equation?tex=P%5Cleft%28Y%3D0%7Cx%5Cright%29%3D%5Cfrac%7B1%7D%7B+1+%2B+exp%5Cleft%28w%5Ccdot+x%2Bb%5Cright%29%7D)

where `w` and `b` are weight and bias, respectively. For convenience, expend weight vector and bias vector, namely,

![\theta = \left(w^{(1)},w^{(1)},...,w^{(n)},b\right)\\ x =  \left(x^{(1)},x^{(1)},...,x^{(n)},1\right)\\](https://www.zhihu.com/equation?tex=%5Ctheta+%3D+%5Cleft%28w%5E%7B%281%29%7D%2Cw%5E%7B%281%29%7D%2C...%2Cw%5E%7B%28n%29%7D%2Cb%5Cright%29%5C%5C+x+%3D++%5Cleft%28x%5E%7B%281%29%7D%2Cx%5E%7B%281%29%7D%2C...%2Cx%5E%7B%28n%29%7D%2C1%5Cright%29%5C%5C)

Then, the binomial logistic regression model is:

![P\left(Y=1|x\right)=\frac{exp\left(\theta^{T} x\right)}{ 1 + exp\left(\theta^{T} x\right)}](https://www.zhihu.com/equation?tex=P%5Cleft%28Y%3D1%7Cx%5Cright%29%3D%5Cfrac%7Bexp%5Cleft%28%5Ctheta%5E%7BT%7D+x%5Cright%29%7D%7B+1+%2B+exp%5Cleft%28%5Ctheta%5E%7BT%7D+x%5Cright%29%7D)

![P\left(Y=0|x\right)=\frac{1}{ 1 + exp\left(\theta^{T} x\right)}](https://www.zhihu.com/equation?tex=P%5Cleft%28Y%3D0%7Cx%5Cright%29%3D%5Cfrac%7B1%7D%7B+1+%2B+exp%5Cleft%28%5Ctheta%5E%7BT%7D+x%5Cright%29%7D)

## Logistic Regression Model

Logistic Regression model consists of parameters estimation, optimization algorithm and classify.

### Parameters Estimation

In [Step-by-Step Guide To Implement Machine Learning III - Naive Bayes](https://www.codeproject.com/Articles/4051340/Step-by-Step-Guide-To-Implement-Machine-Learning-3), we use the Maximum likelihood function to estimate the parameters in  the Baysian model. Similarly, we use Maximum likelihood function to  estimate the parameters in Logistic Regression Model. Denote

![P\left(Y=1|x\right)=\pi_{\theta}\left(x\right) ](https://www.zhihu.com/equation?tex=P%5Cleft%28Y%3D1%7Cx%5Cright%29%3D%5Cpi_%7B%5Ctheta%7D%5Cleft%28x%5Cright%29+)

![P\left(Y=0|x\right)=1-\pi_{\theta}\left(x\right)](https://www.zhihu.com/equation?tex=P%5Cleft%28Y%3D0%7Cx%5Cright%29%3D1-%5Cpi_%7B%5Ctheta%7D%5Cleft%28x%5Cright%29)

where:

![\pi_{\theta}\left(x\right) =g\left(\theta^{T}x\right)=\frac{1}{1+e^{-\theta^{T}x}}](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%5Cleft%28x%5Cright%29+%3Dg%5Cleft%28%5Ctheta%5E%7BT%7Dx%5Cright%29%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-%5Ctheta%5E%7BT%7Dx%7D%7D)

`g(x)` is also called **sigmoid function**. The likehood function is:

![\prod_{i=1}^{N}\left[\pi\left(x^{(i)}\right)\right]^{y^{(i)}}\left[1-\pi\left(x^{(i)}\right)\right]^{1-y^{(i)}}](https://www.zhihu.com/equation?tex=%5Cprod_%7Bi%3D1%7D%5E%7BN%7D%5Cleft%5B%5Cpi%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29%5Cright%5D%5E%7By%5E%7B%28i%29%7D%7D%5Cleft%5B1-%5Cpi%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29%5Cright%5D%5E%7B1-y%5E%7B%28i%29%7D%7D)

For convenience, we take the logarithm of the likehood function, namely:

![L\left(\theta\right)=\sum_{i=1}^{N}\left[ y^{(i)}\log\pi_{\theta}\left(x^{(i)}\right)+\left(1-y^{(i)}\right)\log\left(1-\pi_{\theta}\left(x^{(i)}\right)\right)\right]\\ ](https://www.zhihu.com/equation?tex=L%5Cleft%28%5Ctheta%5Cright%29%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cleft%5B+y%5E%7B%28i%29%7D%5Clog%5Cpi_%7B%5Ctheta%7D%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29%2B%5Cleft%281-y%5E%7B%28i%29%7D%5Cright%29%5Clog%5Cleft%281-%5Cpi_%7B%5Ctheta%7D%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29%5Cright%29%5Cright%5D%5C%5C+)

Then, the problem is transformed into calculating the max of the likehood function.

### Optimization Algorithm

Because, **we cannot get an analytic solutions to the derivative of likehood function**. To get the max of likehood function, we apply the **gradient ascent method**, namely:

![\theta:=\theta+\alpha \nabla_{\theta}L\left(\theta\right)](https://www.zhihu.com/equation?tex=%5Ctheta%3A%3D%5Ctheta%2B%5Calpha+%5Cnabla_%7B%5Ctheta%7DL%5Cleft%28%5Ctheta%5Cright%29)

calculate the derivative of likelihood function:

![\begin{align*}  \frac{\partial }{\partial \theta_{j}}L\left(\theta\right) & = \left(y{\frac{1}{g\left(\theta^Tx\right)}}-\left(1-y\right)\frac{1}{1-g\left(\theta^Tx\right)}\right)\frac{\partial}{\partial\theta_{j}}g\left(\theta^Tx\right)\\ &=\left(y{\frac{1}{g\left(\theta^Tx\right)}}-\left(1-y\right)\frac{1}{1-g\left(\theta^Tx\right)}\right)g\left(\theta^Tx\right)\left(1-g\left(\theta^{T}x\right)\right)\frac{\partial}{\partial\theta_{j}}\theta^Tx\\ &=\left(y\left(1-g\left(\theta^{T}x\right)\right)-\left(1-y\right)g\left(\theta^{T}x\right)\right)x_{j}\\ &=\left(y-\pi_{\theta}\left(x\right)\right)x_{j}  \end{align*}](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D++%5Cfrac%7B%5Cpartial+%7D%7B%5Cpartial+%5Ctheta_%7Bj%7D%7DL%5Cleft%28%5Ctheta%5Cright%29+%26+%3D+%5Cleft%28y%7B%5Cfrac%7B1%7D%7Bg%5Cleft%28%5Ctheta%5ETx%5Cright%29%7D%7D-%5Cleft%281-y%5Cright%29%5Cfrac%7B1%7D%7B1-g%5Cleft%28%5Ctheta%5ETx%5Cright%29%7D%5Cright%29%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%5Ctheta_%7Bj%7D%7Dg%5Cleft%28%5Ctheta%5ETx%5Cright%29%5C%5C+%26%3D%5Cleft%28y%7B%5Cfrac%7B1%7D%7Bg%5Cleft%28%5Ctheta%5ETx%5Cright%29%7D%7D-%5Cleft%281-y%5Cright%29%5Cfrac%7B1%7D%7B1-g%5Cleft%28%5Ctheta%5ETx%5Cright%29%7D%5Cright%29g%5Cleft%28%5Ctheta%5ETx%5Cright%29%5Cleft%281-g%5Cleft%28%5Ctheta%5E%7BT%7Dx%5Cright%29%5Cright%29%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%5Ctheta_%7Bj%7D%7D%5Ctheta%5ETx%5C%5C+%26%3D%5Cleft%28y%5Cleft%281-g%5Cleft%28%5Ctheta%5E%7BT%7Dx%5Cright%29%5Cright%29-%5Cleft%281-y%5Cright%29g%5Cleft%28%5Ctheta%5E%7BT%7Dx%5Cright%29%5Cright%29x_%7Bj%7D%5C%5C+%26%3D%5Cleft%28y-%5Cpi_%7B%5Ctheta%7D%5Cleft%28x%5Cright%29%5Cright%29x_%7Bj%7D++%5Cend%7Balign%2A%7D)

Let the derivative equal to zero, we can get:

![\theta := \theta+\alpha\sum_{i=1}^{m}\left(y^{(i)}-\pi_{\theta}\left(x^{(i)}\right)\right)x^{(i)}_{j}](https://www.zhihu.com/equation?tex=%5Ctheta+%3A%3D+%5Ctheta%2B%5Calpha%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Cleft%28y%5E%7B%28i%29%7D-%5Cpi_%7B%5Ctheta%7D%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29%5Cright%29x%5E%7B%28i%29%7D_%7Bj%7D)

Thus, we can get the optimized parameter through the above equation. The code of gradient ascent method is shown below:

Hide   Copy Code

```
if method == "GA":
weights = np.random.normal(0, 1, [feature_dim, 1])
for i in range(iterations):
    pred = self.sigmoid(np.dot(train_data, weights))
    errors = train_label - pred
    # update the weights
    weights = weights + alpha * np.dot(train_data.T, errors)
self.weights = weights
return self
```

### Classify

In logistics regression model, sigmoid function is applied to calculate the probability, which is expressed as:

![sigmoid\left(x\right)=\frac{1}{1+e^{-x}}](https://www.zhihu.com/equation?tex=sigmoid%5Cleft%28x%5Cright%29%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-x%7D%7D)

When the result is larger than `0.5`, the sample belongs to class 1, else it belongs to class `0`.

Hide   Copy Code

```
def sigmoid(self, x, derivative=False):
    output = 1/(1 + np.exp(-x))
    if derivative:
       output = output * (1 - output)
    return output
```

## Conclusion and Analysis

To get the parameters of the logistic regression model, we can also  minimize the loss function. Finally, let's compare our logistics  regression with the Sklearn's and the detection performance is displayed below:

![Image 15](https://www.codeproject.com/KB/AI/4061324/aeda2f20-5e1c-4907-a6dc-63c0eeead315.Png)

The detection performance of both is similar.

The related code and dataset in this article can be found in [MachineLearning](https://github.com/DandelionLau/MachineLearning).