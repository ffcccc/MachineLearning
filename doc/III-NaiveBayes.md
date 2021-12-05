# Step-by-Step Guide To Implement Machine Learning III - Naive Bayes


## Introduction

Naive Bayes is a kind of classification based on Bayesian decision  theory and feature conditional independence, which calculates the  probability distribution based on conditional independence on training  set as the detection model. For a given test object, the label of the  maximum of the posterior probability is the prediction of the test  object. Maximize the posterior probability means minimizing the expected risk. Then another question is why call it "**Naive**" Bayes? This is because Naive Bayes follow such a naive hypothesis: **all the features for classification are independent when the label is definitized**, which is given by:

![P\left(X = x| Y = c_{k}\right)=P\left(X^{\left(1\right)}=x^{\left(1\right)},...,X^{\left(n\right)}|Y= c_{k}\right)=\prod_{j=1}^{n}P\left( X^{\left(j\right)}=x^{\left(j\right)}|Y=c_{k}\right)](https://www.zhihu.com/equation?tex=P%5Cleft%28X+%3D+x%7C+Y+%3D+c_%7Bk%7D%5Cright%29%3DP%5Cleft%28X%5E%7B%5Cleft%281%5Cright%29%7D%3Dx%5E%7B%5Cleft%281%5Cright%29%7D%2C...%2CX%5E%7B%5Cleft%28n%5Cright%29%7D%7CY%3D+c_%7Bk%7D%5Cright%29%3D%5Cprod_%7Bj%3D1%7D%5E%7Bn%7DP%5Cleft%28+X%5E%7B%5Cleft%28j%5Cright%29%7D%3Dx%5E%7B%5Cleft%28j%5Cright%29%7D%7CY%3Dc_%7Bk%7D%5Cright%29)

where *x(j)* is the i-th feature, *ck* is the `*k*-th` label. Then, the Bayes classifier can be defined as:

![y =arg\max \limits_{c_{k}}P\left(Y=c_{k}\right)\prod_{j}P\left(X^{\left(j\right)}=x^{\left(j\right)}|Y=c_{k}\right) ](https://www.zhihu.com/equation?tex=y+%3Darg%5Cmax+%5Climits_%7Bc_%7Bk%7D%7DP%5Cleft%28Y%3Dc_%7Bk%7D%5Cright%29%5Cprod_%7Bj%7DP%5Cleft%28X%5E%7B%5Cleft%28j%5Cright%29%7D%3Dx%5E%7B%5Cleft%28j%5Cright%29%7D%7CY%3Dc_%7Bk%7D%5Cright%29+)

So why maximize the posterior probability means minimizing the expected risk ? Let the loss is 0-1 loss function is

![L\left(Y,f\left(X\right)\right)=\left\{\begin{aligned} 0,Y\ne f\left(X\right)\\ 1,Y=f\left(X\right) \end{aligned}\right.](https://www.zhihu.com/equation?tex=L%5Cleft%28Y%2Cf%5Cleft%28X%5Cright%29%5Cright%29%3D%5Cleft%5C%7B%5Cbegin%7Baligned%7D+0%2CY%5Cne+f%5Cleft%28X%5Cright%29%5C%5C+1%2CY%3Df%5Cleft%28X%5Cright%29+%5Cend%7Baligned%7D%5Cright.)

where `f(x)` is the decision function. Then, the expected risk is

![R_{exp}\left(f\right)=E\left[L\left(Y,f\left(X\right)\right)\right]](https://www.zhihu.com/equation?tex=R_%7Bexp%7D%5Cleft%28f%5Cright%29%3DE%5Cleft%5BL%5Cleft%28Y%2Cf%5Cleft%28X%5Cright%29%5Cright%29%5Cright%5D)

which is calculated from joint distribution `P(X,Y)`. Thus the conditional expectation is:

![R_{exp}\left(f\right)=E_x\sum_{k=1}^{K}\left[L\left(c_{k},f\left(X\right)\right)\right]P\left(c_{k}|X\right)](https://www.zhihu.com/equation?tex=R_%7Bexp%7D%5Cleft%28f%5Cright%29%3DE_x%5Csum_%7Bk%3D1%7D%5E%7BK%7D%5Cleft%5BL%5Cleft%28c_%7Bk%7D%2Cf%5Cleft%28X%5Cright%29%5Cright%29%5Cright%5DP%5Cleft%28c_%7Bk%7D%7CX%5Cright%29)

To minimize the expected risk, it needs to minimize each `*X = x*`, namely:

![f\left(x\right) =arg\min\limits_{y\in Y}\sum_{k=1}^{K}L\left(c_{k},y\right)P\left(c_{k}|X=x\right)\\ =arg\min\limits_{y\in Y} \sum_{k=1}^{K}P\left(y \ne c_{k}|X=x\right)\\ =arg\min\limits_{y\in Y}\left(1-P\left(y = c_{k}|X=x\right)\right)\\ =arg\min\limits_{y\in Y}P\left(y = c_{k}|X=x\right)](https://www.zhihu.com/equation?tex=f%5Cleft%28x%5Cright%29+%3Darg%5Cmin%5Climits_%7By%5Cin+Y%7D%5Csum_%7Bk%3D1%7D%5E%7BK%7DL%5Cleft%28c_%7Bk%7D%2Cy%5Cright%29P%5Cleft%28c_%7Bk%7D%7CX%3Dx%5Cright%29%5C%5C+%3Darg%5Cmin%5Climits_%7By%5Cin+Y%7D+%5Csum_%7Bk%3D1%7D%5E%7BK%7DP%5Cleft%28y+%5Cne+c_%7Bk%7D%7CX%3Dx%5Cright%29%5C%5C+%3Darg%5Cmin%5Climits_%7By%5Cin+Y%7D%5Cleft%281-P%5Cleft%28y+%3D+c_%7Bk%7D%7CX%3Dx%5Cright%29%5Cright%29%5C%5C+%3Darg%5Cmin%5Climits_%7By%5Cin+Y%7DP%5Cleft%28y+%3D+c_%7Bk%7D%7CX%3Dx%5Cright%29)

## Naive Bayes Model

Naive Bayes model consists of parameters estimation and classify.

### Parameters Estimation

In the training process, learning means estimate the prior probability and conditional probability. **Maximum likelihood estimation** (MLE) is a general method to get the above parameters. The MLE of prior probability is given by:

![P\left( Y=c_{k}\right)=\frac{\sum_{i=1}^{N}{I\left(y_{i}=c_{k}\right)}}{N}](https://www.zhihu.com/equation?tex=P%5Cleft%28+Y%3Dc_%7Bk%7D%5Cright%29%3D%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7BI%5Cleft%28y_%7Bi%7D%3Dc_%7Bk%7D%5Cright%29%7D%7D%7BN%7D)

Denote the` ``j-th` feature set is` `{*aj1*,*aj2*,...,*ajsi*}.Then, the MLE of conditional probability is given by:

![P_{\lambda}\left(X^{(j)}=a_{jl}|Y=c_{k}\right)=\frac{\sum_{i=1}^{N}I\left(x_{i}^{(j)}=a_{jl},y_{i}=c_{k}\right)}{\sum_{i=1}^{N}I\left(y_{i}=c_{k}\right)}](https://www.zhihu.com/equation?tex=P_%7B%5Clambda%7D%5Cleft%28X%5E%7B%28j%29%7D%3Da_%7Bjl%7D%7CY%3Dc_%7Bk%7D%5Cright%29%3D%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7DI%5Cleft%28x_%7Bi%7D%5E%7B%28j%29%7D%3Da_%7Bjl%7D%2Cy_%7Bi%7D%3Dc_%7Bk%7D%5Cright%29%7D%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7DI%5Cleft%28y_%7Bi%7D%3Dc_%7Bk%7D%5Cright%29%7D)

In the Naive Bayes training process, the prior probability and  conditional probability is calculated. However, if a value of a feature  has never occurred in the training set, it's probability is equal to  zero, which will effect the result of posterior probability. To solve  the problem, we introduce **Laplace smoothing**: **add an integer ![\lambda](https://www.zhihu.com/equation?tex=%5Clambda) to the frequency of each random variable**.

Then, the Bayesian estimation of prior probability is:

![P_{\lambda}\left( Y=c_{k}\right)=\frac{\sum_{i=1}^{N}{I\left(y_{i}=c_{k}\right)+\lambda}}{N+K\lambda}](https://www.zhihu.com/equation?tex=P_%7B%5Clambda%7D%5Cleft%28+Y%3Dc_%7Bk%7D%5Cright%29%3D%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7BI%5Cleft%28y_%7Bi%7D%3Dc_%7Bk%7D%5Cright%29%2B%5Clambda%7D%7D%7BN%2BK%5Clambda%7D)

where `*N*` is the number of unique labels, the `*K* `is the number of samples. The code of prior probability is shown below:

Hide   Copy Code

```
prior_probability = {}
for key in range(len(label_value)):
  prior_probability[label_value[key][0]] = 
    (label_value[key][1] + self.laplace) / (N + K * self.laplace)  # laplace smooth
self.prior_probability = prior_probability
```

where ` label_value ` is the tuple of ` (label, label_num)`.

Similarly, the Bayesian estimation of conditional probability is:

![P_{\lambda}\left(X^{(j)}=a_{jl}|Y=c_{k}\right)=\frac{\sum_{i=1}^{N}I\left(x_{i}^{(j)}=a_{jl},y_{i}=c_{k}\right)+\lambda}{\sum_{i=1}^{N}I\left(y_{i}=c_{k}\right)+S_{j}\lambda}](https://www.zhihu.com/equation?tex=P_%7B%5Clambda%7D%5Cleft%28X%5E%7B%28j%29%7D%3Da_%7Bjl%7D%7CY%3Dc_%7Bk%7D%5Cright%29%3D%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7DI%5Cleft%28x_%7Bi%7D%5E%7B%28j%29%7D%3Da_%7Bjl%7D%2Cy_%7Bi%7D%3Dc_%7Bk%7D%5Cright%29%2B%5Clambda%7D%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7DI%5Cleft%28y_%7Bi%7D%3Dc_%7Bk%7D%5Cright%29%2BS_%7Bj%7D%5Clambda%7D)

The code of conditional probability is shown below. A matrix is applied to save the conditional probability and` S[j]` is the number of unique labels of the `j-th` feature.

Hide   Copy Code

```
# calculate the conditional probability
prob = []
# calculate the count (x = a & y = c)
for j in range(feature_dim):
    count = np.zeros([S[j], len(label_count)])  # the range of label start with 1
    feature_temp = train_data[:, j]
    feature_value_temp = feature_value[j]
    for i in range(len(feature_temp)):
        for k in range(len(feature_value_temp)):
            for t in range(len(label_count)):
                if feature_temp[i] == feature_value_temp[k]
                        and train_label[i] == label_value[t][0]:
                   count[k][t] += 1             # x = value and y = label
     # calculate the conditional probability
     for m in range(len(label_value)):
         count[:, m] = (count[:, m] + self.laplace) /
                 (label_value[m][1] + self.laplace*S[j])  # laplace smoothing
         # print(count)
    prob.append(count)
self.conditional_probability = prob
```

### Classify

After calculating the prior probability and conditional probability, the Bayesian classification model is:

![y =arg\max \limits_{c_{k}}P\left(Y=c_{k}\right)\prod_{j}P\left(X^{\left(j\right)}=x^{\left(j\right)}|Y=c_{k}\right) ](https://www.zhihu.com/equation?tex=y+%3Darg%5Cmax+%5Climits_%7Bc_%7Bk%7D%7DP%5Cleft%28Y%3Dc_%7Bk%7D%5Cright%29%5Cprod_%7Bj%7DP%5Cleft%28X%5E%7B%5Cleft%28j%5Cright%29%7D%3Dx%5E%7B%5Cleft%28j%5Cright%29%7D%7CY%3Dc_%7Bk%7D%5Cright%29+)

The classification code is shown below. The `predict` is a dictionary which includes the probability of each label. Then we just need to sort the `predict `and the prediction is the first element in the sorted dictionary.

Hide   Copy Code

```
def classify(self, sample):
    predict = {}
    for m in range(len(self.label_value)):
        temp = self.prior_probability
          [self.label_value[m][0]]  # get the prior_probability of m-th label in label_value
        for n in range(len(sample)):
            if sample[n] in self.feature_value[n]:
                # print(m, n)
                index = np.where(self.feature_value[n] == sample[n])[0][0]
                temp = temp * self.conditional_probability[n][index][m]
            else:
                temp = self.laplace /
                     (self.S[n] * self.laplace)  # if the value of feature is
                                    # not in training set, return the laplace smoothing
        predict[self.label_value[m][0]] = temp
    return predict
```

## Conclusion and Analysis

The Bayesian model is this article is Berniulli Bayesian model.  Except that, there are other Bayesian model such as Guassian Bayesian  model, Polynomial Bayesian model. Finally, let's compare our Bayesian  model with the Bayes model in Sklearn and the detection performance is  displayed below:

![Image 13](https://www.codeproject.com/KB/AI/4051340/d8fac4a5-fb08-457c-8857-aae12b082d20.Png)

It is found that both methods achieve poor detection results.  Moreover, our Bayesian model takes a longer runtime, which may be that  the algorithm of conditional probability contains too many loops.

The related code and dataset in this article can be found in [MachineLearning](https://github.com/DandelionLau/MachineLearning).