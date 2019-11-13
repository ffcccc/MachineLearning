# Step-by-Step Guide to Implement Machine Learning VIII - Linear Regression

![img](https://www.codeproject.com/script/Membership/ProfileImages/{6098a88e-6a47-4a71-915c-4577a7b84ee9}.jpg)

[danc1elion](https://www.codeproject.com/script/Membership/View.aspx?mid=14354398)

|      | Rate this: 			 				 			                      				   	 	         ![img](https://codeproject.freetls.fastly.net/script/Ratings/Images/stars-fill-lg.png)![img](https://codeproject.freetls.fastly.net/script/Ratings/Images/stars-empty-lg.png) 		  	 	5.00  (2 votes) |
| ---- | ------------------------------------------------------------ |
|      |                                                              |

​                                        27 May 2019[CPOL](http://www.codeproject.com/info/cpol10.aspx)                                    

Easy to implement machine learning



This article is an entry in our [Machine Learning and Artificial Intelligence Challenge](https://www.codeproject.com/Competitions/1024/The-Machine-Learning-and-Artificial-Intelligence-C.aspx). Articles in this sub-section are not required to be full articles so care should be taken when voting.

## Introduction

There universally exists a relationship among variables. Indeed, the  relationship can be divided into two categories, namely, certainty  relation and uncertainty relation. The certainty relation can be  expressed with a function. The certainty relation is also called  correlation, which can be studied with regression analysis.

Generally, the linear regression model is:

![h_{\theta}\left(x\right)=\theta^{T}x](https://www.zhihu.com/equation?tex=h_%7B%5Ctheta%7D%5Cleft%28x%5Cright%29%3D%5Ctheta%5E%7BT%7Dx)

The optimal ![\theta](https://www.zhihu.com/equation?tex=%5Ctheta) can be determined by minimum the loss function:

![J\left(\theta\right)=\sum^{m}_{i=1}\left(h_{\theta}\left(x\right)^{(i)}-y^{(i)}\right)^{2}](https://www.zhihu.com/equation?tex=J%5Cleft%28%5Ctheta%5Cright%29%3D%5Csum%5E%7Bm%7D_%7Bi%3D1%7D%5Cleft%28h_%7B%5Ctheta%7D%5Cleft%28x%5Cright%29%5E%7B%28i%29%7D-y%5E%7B%28i%29%7D%5Cright%29%5E%7B2%7D)

## Regression Model

Linear regression consists of linear regression, local weighted  linear regression, ridge regression, Lasso regression and stepwise  linear regression.

### Linear Regression

The parameter ![\theta](https://www.zhihu.com/equation?tex=%5Ctheta) for linear regression can be calculated by gradient descent method or **regular expression**. Because gradient descent method has been introduced in [Step-by-Step Guide to Implement Machine Learning IV - Logistic Regression](https://www.codeproject.com/Articles/4061324/Step-by-Step-Guide-to-Implement-Machine-Learning-4), we introduce the solution with regular expression in this article.

First, calculate the derivative of loss function:

![\begin{align*} \nabla_{\theta}J\left(\theta\right)& =\nabla_{\theta}\frac{1}{2}\left(X\theta-Y\right)^{T}\left(X\theta-Y\right)\\ &= \frac{1}{2}\nabla_{\theta}\left(\theta^{T}X^{T}X\theta-\theta^{T}X^{T}Y-YX\theta+Y^{T}Y\right)\\ &=\frac{1}{2}\nabla_{\theta}tr\left(\theta^{T}X^{T}X\theta-\theta^{T}X^{T}Y-YX\theta+Y^{T}Y\right)\\ &=\frac{1}{2}\nabla_{\theta}\left(tr\theta^{T}X^{T}X\theta-2trYX\theta\right)\\ &=\frac{1}{2}\nabla_{\theta}\left(X^{T}X\theta+X^{T}X\theta-2X^{T}Y\right)\\ &= X^{T}X\theta-X^{T}Y \end{align*}](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D+%5Cnabla_%7B%5Ctheta%7DJ%5Cleft%28%5Ctheta%5Cright%29%26+%3D%5Cnabla_%7B%5Ctheta%7D%5Cfrac%7B1%7D%7B2%7D%5Cleft%28X%5Ctheta-Y%5Cright%29%5E%7BT%7D%5Cleft%28X%5Ctheta-Y%5Cright%29%5C%5C+%26%3D+%5Cfrac%7B1%7D%7B2%7D%5Cnabla_%7B%5Ctheta%7D%5Cleft%28%5Ctheta%5E%7BT%7DX%5E%7BT%7DX%5Ctheta-%5Ctheta%5E%7BT%7DX%5E%7BT%7DY-YX%5Ctheta%2BY%5E%7BT%7DY%5Cright%29%5C%5C+%26%3D%5Cfrac%7B1%7D%7B2%7D%5Cnabla_%7B%5Ctheta%7Dtr%5Cleft%28%5Ctheta%5E%7BT%7DX%5E%7BT%7DX%5Ctheta-%5Ctheta%5E%7BT%7DX%5E%7BT%7DY-YX%5Ctheta%2BY%5E%7BT%7DY%5Cright%29%5C%5C+%26%3D%5Cfrac%7B1%7D%7B2%7D%5Cnabla_%7B%5Ctheta%7D%5Cleft%28tr%5Ctheta%5E%7BT%7DX%5E%7BT%7DX%5Ctheta-2trYX%5Ctheta%5Cright%29%5C%5C+%26%3D%5Cfrac%7B1%7D%7B2%7D%5Cnabla_%7B%5Ctheta%7D%5Cleft%28X%5E%7BT%7DX%5Ctheta%2BX%5E%7BT%7DX%5Ctheta-2X%5E%7BT%7DY%5Cright%29%5C%5C+%26%3D+X%5E%7BT%7DX%5Ctheta-X%5E%7BT%7DY+%5Cend%7Balign%2A%7D)

Then, make the derivative equal to 0, we can obtain:

![X^{T}X\theta=X^{T}Y](https://www.zhihu.com/equation?tex=X%5E%7BT%7DX%5Ctheta%3DX%5E%7BT%7DY)

Finally, ![\theta](https://www.zhihu.com/equation?tex=%5Ctheta) is:

![\theta=\left(X^{T}X\right)^{-1}X^{T}Y](https://www.zhihu.com/equation?tex=%5Ctheta%3D%5Cleft%28X%5E%7BT%7DX%5Cright%29%5E%7B-1%7DX%5E%7BT%7DY)

where X is the training data and Y is the corresponding label. The code of linear regression is shown below:

Hide   Copy Code

```
def standardLinearRegression(self, x, y):
    if self.norm_type == "Standardization":
        x = preProcess.Standardization(x)
    else:
        x = preProcess.Normalization(x)

    xTx = np.dot(x.T, x)
    if np.linalg.det(xTx) == 0:   # calculate the Determinant of xTx
        print("Error: Singluar Matrix !")
        return
    w = np.dot(np.linalg.inv(xTx), np.dot(x.T, y))
    return w
```

### Local Weighted Linear Regression

It is underfitting in linear regression for it using the unbiased  estimation of minimum mean square error(MMSE). To solve the problem, we  assign weights on the points around the point to be predicted. Then, we  apply normal regression analysis on it. The loss function for local  weighted linear regression is:

![J\left(\theta\right)=\sum^{m}_{i=1}w^{(i)}\left(h_{\theta}\left(x\right)^{(i)}-y^{(i)}\right)^{2}](https://www.zhihu.com/equation?tex=J%5Cleft%28%5Ctheta%5Cright%29%3D%5Csum%5E%7Bm%7D_%7Bi%3D1%7Dw%5E%7B%28i%29%7D%5Cleft%28h_%7B%5Ctheta%7D%5Cleft%28x%5Cright%29%5E%7B%28i%29%7D-y%5E%7B%28i%29%7D%5Cright%29%5E%7B2%7D)

Like linear regression, we calculate the derivative of loss function and make it equal to 0. The optimal ![\theta](https://www.zhihu.com/equation?tex=%5Ctheta) is

![\theta=\left(X^{T}WX\right)^{-1}X^{T}WY](https://www.zhihu.com/equation?tex=%5Ctheta%3D%5Cleft%28X%5E%7BT%7DWX%5Cright%29%5E%7B-1%7DX%5E%7BT%7DWY)

The weights in local weighted linear regression is like the kernel function in SVM, which is given by:

![w^{(i)}=exp\left(-\frac{\left(x^{(i)}-x\right)^{T}\left(x^{(i)}-x\right)}{2\tau^{2}}\right)](https://www.zhihu.com/equation?tex=w%5E%7B%28i%29%7D%3Dexp%5Cleft%28-%5Cfrac%7B%5Cleft%28x%5E%7B%28i%29%7D-x%5Cright%29%5E%7BT%7D%5Cleft%28x%5E%7B%28i%29%7D-x%5Cright%29%7D%7B2%5Ctau%5E%7B2%7D%7D%5Cright%29)

The code of local weighted linear regression is shown below:

Hide   Copy Code

```
def LWLinearRegression(self, x, y, sample):
    if self.norm_type == "Standardization":
        x = preProcess.Standardization(x)
    else:
        x = preProcess.Normalization(x)

    sample_num = len(x)
    weights = np.eye(sample_num)
    for i in range(sample_num):
        diff = sample - x[i, :]
        weights[i, i] = np.exp(np.dot(diff, diff.T)/(-2 * self.k ** 2))
    xTx = np.dot(x.T, np.dot(weights, x))
    if np.linalg.det(xTx) == 0:
        print("Error: Singluar Matrix !")
        return
    result = np.dot(np.linalg.inv(xTx), np.dot(x.T, np.dot(weights, y)))
    return np.dot(sample.T, result)
```

### Ridge Regression

If the feature dimension is large, than the number of samples, the  input matrix is not full rank, whose inverse matrix doesn't exist. To  solve the problem, ridge regression add ![\lambda I](https://www.zhihu.com/equation?tex=%5Clambda+I) to make the matrix nonsingular. Actually, it is equal to add **L2 regularization** on the loss function for ridge regression, namely:

![J\left(\theta\right)=\sum^{m}_{i=1}\left(h_{\theta}\left(x\right)^{(i)}-y^{(i)}\right)^{2}+\lambda\left| |\theta| \right|_{2}](https://www.zhihu.com/equation?tex=J%5Cleft%28%5Ctheta%5Cright%29%3D%5Csum%5E%7Bm%7D_%7Bi%3D1%7D%5Cleft%28h_%7B%5Ctheta%7D%5Cleft%28x%5Cright%29%5E%7B%28i%29%7D-y%5E%7B%28i%29%7D%5Cright%29%5E%7B2%7D%2B%5Clambda%5Cleft%7C+%7C%5Ctheta%7C+%5Cright%7C_%7B2%7D)

Like linear regression, we calculate the derivative of loss function and make it equal to 0. The optimal ![\theta](https://www.zhihu.com/equation?tex=%5Ctheta) is:

![\theta=\left(X^{T}X+\lambda^{2}I\right)^{-1}X^{T}Y](https://www.zhihu.com/equation?tex=%5Ctheta%3D%5Cleft%28X%5E%7BT%7DX%2B%5Clambda%5E%7B2%7DI%5Cright%29%5E%7B-1%7DX%5E%7BT%7DY)

The code of ridge regression is shown below:

Hide   Copy Code

```
def ridgeRegression(self, x, y):
    if self.norm_type == "Standardization":
        x = preProcess.Standardization(x)
    else:
        x = preProcess.Normalization(x)

    feature_dim = len(x[0])
    xTx = np.dot(x.T, x)
    matrix = xTx + np.exp(feature_dim)*self.lamda
    if np.linalg.det(xTx) == 0:
        print("Error: Singluar Matrix !")
        return
    w = np.dot(np.linalg.inv(matrix), np.dot(x.T, y))
    return w
```

### Lasso Regression

Like ridge regression, Lasso regression add **L1 regularization** on the loss function, namely:

![J\left(\theta\right)=\sum^{m}_{i=1}\left(h_{\theta}\left(x\right)^{(i)}-y^{(i)}\right)^{2}+\lambda\left| |\theta| \right|_{1}](https://www.zhihu.com/equation?tex=J%5Cleft%28%5Ctheta%5Cright%29%3D%5Csum%5E%7Bm%7D_%7Bi%3D1%7D%5Cleft%28h_%7B%5Ctheta%7D%5Cleft%28x%5Cright%29%5E%7B%28i%29%7D-y%5E%7B%28i%29%7D%5Cright%29%5E%7B2%7D%2B%5Clambda%5Cleft%7C+%7C%5Ctheta%7C+%5Cright%7C_%7B1%7D)

Because the L1 regularization contains absolute value expression, the loss function is not derivable anywhere. Thus, we apply **coordinate descent method** (CD). The CD gets a minimum at a direction each iteration, namely,

![\theta^{i+1}_{j}=arg\min\limits_{\theta_{i}} J\left(\theta_{1},\theta_{2}^{(i-1)},..., \theta_{n}^{(i-1)}\right)](https://www.zhihu.com/equation?tex=%5Ctheta%5E%7Bi%2B1%7D_%7Bj%7D%3Darg%5Cmin%5Climits_%7B%5Ctheta_%7Bi%7D%7D+J%5Cleft%28%5Ctheta_%7B1%7D%2C%5Ctheta_%7B2%7D%5E%7B%28i-1%29%7D%2C...%2C+%5Ctheta_%7Bn%7D%5E%7B%28i-1%29%7D%5Cright%29)

We can get a closed solution for CD, which is given by:

![\begin{equation} \left\{              \begin{array}{lr}              \theta_{j}=\rho_{j}+\lambda &\ if\ \rho_{i}<-\lambda  \\           \theta_{j}=0 &if\ -\lambda\leq\rho_{i}\leq\lambda \\ \theta_{j}=\rho_{j}-\lambda &\ if\ \rho_{i}>\lambda  \\              \end{array} \right. \end{equation} ](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%5Cleft%5C%7B++++++++++++++%5Cbegin%7Barray%7D%7Blr%7D++++++++++++++%5Ctheta_%7Bj%7D%3D%5Crho_%7Bj%7D%2B%5Clambda+%26%5C+if%5C+%5Crho_%7Bi%7D%3C-%5Clambda++%5C%5C+++++++++++%5Ctheta_%7Bj%7D%3D0+%26if%5C+-%5Clambda%5Cleq%5Crho_%7Bi%7D%5Cleq%5Clambda+%5C%5C+%5Ctheta_%7Bj%7D%3D%5Crho_%7Bj%7D-%5Clambda+%26%5C+if%5C+%5Crho_%7Bi%7D%3E%5Clambda++%5C%5C++++++++++++++%5Cend%7Barray%7D+%5Cright.+%5Cend%7Bequation%7D+)

where:

![\rho_{j}=\sum_{i=1}^{m}x_{j}^{(i)}\left(y^{(i)}-\sum_{k\ne j}^{n}\theta_{k}x_{k}^{(i)}\right)\\=\sum_{i=1}^{m}x_{j}^{(i)}\left(y^{(i)}-\hat y_{pred}^{(i)}+\theta_{j}x^{(i)}_{j}\right)](https://www.zhihu.com/equation?tex=%5Crho_%7Bj%7D%3D%5Csum_%7Bi%3D1%7D%5E%7Bm%7Dx_%7Bj%7D%5E%7B%28i%29%7D%5Cleft%28y%5E%7B%28i%29%7D-%5Csum_%7Bk%5Cne+j%7D%5E%7Bn%7D%5Ctheta_%7Bk%7Dx_%7Bk%7D%5E%7B%28i%29%7D%5Cright%29%5C%5C%3D%5Csum_%7Bi%3D1%7D%5E%7Bm%7Dx_%7Bj%7D%5E%7B%28i%29%7D%5Cleft%28y%5E%7B%28i%29%7D-%5Chat+y_%7Bpred%7D%5E%7B%28i%29%7D%2B%5Ctheta_%7Bj%7Dx%5E%7B%28i%29%7D_%7Bj%7D%5Cright%29)

The code of Lasso regression is shown below:

Hide   Copy Code

```
def lassoRegression(self, x, y):
    if self.norm_type == "Standardization":
        x = preProcess.Standardization(x)
    else:
        x = preProcess.Normalization(x)

    y = np.expand_dims(y, axis=1)
    sample_num, feataure_dim = np.shape(x)
    w = np.ones([feataure_dim, 1])
    for i in range(self.iterations):
        for j in range(feataure_dim):
            h = np.dot(x[:, 0:j], w[0:j]) + np.dot(x[:, j+1:], w[j+1:])
            w[j] = np.dot(x[:, j], (y - h))
            if j == 0:
                w[j] = 0
            else:
                w[j] = self.softThreshold(w[j])
    return w
```

### Stepwise Linear Regression

Stepwise linear regression is similar to Lasso, which applies greedy  algorithm at each iteration to get a minimum rather than CD. Stepwise  linear regression adds or cuts down a small part on the weights at each  iteration. The code of stepwise linear regression is shown below:

Hide   Copy Code

```
def forwardstepRegression(self, x, y):
    if self.norm_type == "Standardization":
        x = preProcess.Standardization(x)
    else:
        x = preProcess.Normalization(x)

    sample_num, feature_dim = np.shape(x)
    w = np.zeros([self.iterations, feature_dim])
    best_w = np.zeros([feature_dim, 1])
    for i in range(self.iterations):
        min_error = np.inf
        for j in range(feature_dim):
            for sign in [-1, 1]:
                temp_w = best_w
                temp_w[j] += sign * self.learning_rate
                y_hat = np.dot(x, temp_w)
                error = ((y - y_hat) ** 2).sum()           # MSE
                if error < min_error:                   # save the best parameters
                    min_error = error
                    best_w = temp_w
        w = best_w
    return w
```

## Conclusion and Analysis

There are many solutions to get the optimal parameter for linear  regression. In this article, we only introduce some basic algorithms.  Finally, let's compare our linear regression with the linear regression  in Sklearn and the detection performance is displayed below:

Sklearn linear regression performance:

![Image 21](https://www.codeproject.com/KB/AI/5061034/2aced3d2-f8d7-4028-815b-28e9dc7d6639.Png)

Our linear regression performance:

![Image 22](https://www.codeproject.com/KB/AI/5061034/685e8b16-a672-4cdb-b183-7878600ff372.Png)

![Image 23](https://www.codeproject.com/KB/AI/5061034/96a0c1d6-7b40-4891-9adb-cd62b5ba90bd.Png)

The performances look similar.

The related code and dataset in this article can be found in [MachineLearning](https://github.com/DandelionLau/MachineLearning).