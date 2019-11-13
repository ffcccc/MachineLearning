# Step-by-Step Guide to Implement Machine Learning V - Support Vector Machine

![img](https://www.codeproject.com/script/Membership/ProfileImages/{6098a88e-6a47-4a71-915c-4577a7b84ee9}.jpg)

[danc1elion](https://www.codeproject.com/script/Membership/View.aspx?mid=14354398)

|      | Rate this: 			 				 			                      				   	 	         ![img](https://codeproject.freetls.fastly.net/script/Ratings/Images/stars-fill-lg.png)![img](https://codeproject.freetls.fastly.net/script/Ratings/Images/stars-empty-lg.png) 		  	 	5.00  (2 votes) |
| ---- | ------------------------------------------------------------ |
|      |                                                              |

​                                        14 May 2019[CPOL](http://www.codeproject.com/info/cpol10.aspx)                                    

Easy to implement machine learning



This article is an entry in our [Machine Learning and Artificial Intelligence Challenge](https://www.codeproject.com/Competitions/1024/The-Machine-Learning-and-Artificial-Intelligence-C.aspx). Articles in this sub-section are not required to be full articles so care should be taken when voting.

## Introduction

Support Vector Machine(SVM) is a classifier based on the max margin  in feature space. The learning strategy of SVM is to make the margin  max, which can be transformed into a **convex quadratic programming** problem. Before learning the algorithm of SVM, let me introduce some terms.

**Functional margin** is defined as: For a given training set `T`, and hyper-plane`(w,b)`, the functional margin between the hyper-plane `(w,b) `and the sample`(xi, yi)` is:

![\hat{\gamma}_{i}=y_{i}\left(w\cdot x+b\right)](https://www.zhihu.com/equation?tex=%5Chat%7B%5Cgamma%7D_%7Bi%7D%3Dy_%7Bi%7D%5Cleft%28w%5Ccdot+x%2Bb%5Cright%29)

The functional margin between hyper-plane `(w,b)` and the training set *T* is the minimum of ![\hat{\gamma}_{i}](https://www.zhihu.com/equation?tex=%5Chat%7B%5Cgamma%7D_%7Bi%7D)

![\hat{\gamma}=\min\limits_{i=1,...,N} \hat{\gamma}_{i}](https://www.zhihu.com/equation?tex=%5Chat%7B%5Cgamma%7D%3D%5Cmin%5Climits_%7Bi%3D1%2C...%2CN%7D+%5Chat%7B%5Cgamma%7D_%7Bi%7D)

Functional margin indicates the confidence level of the classification results. If the hyper-parameters `(w,b) `change in proportion, for example, change `(w,b)` into `(2w,2b)`. Though the hyper-plane hasn't changed, the functional margin expend doubles. Thus, we apply some contrains on `w `to make the hyper-plane definitized , such as normalization `||w|| = 1`. Then, the margin is called **geometric margin**: For a given training set `T`, and hyper-plane`(w,b)`, the functional margin between the hyper-plane `(w,b) `and the sample`(xi, yi)` is:

![{\gamma}_{i}=y_{i}\left(\frac{w}{||w||}\cdot x_{i}+\frac{b}{||w||}\right)](https://www.zhihu.com/equation?tex=%7B%5Cgamma%7D_%7Bi%7D%3Dy_%7Bi%7D%5Cleft%28%5Cfrac%7Bw%7D%7B%7C%7Cw%7C%7C%7D%5Ccdot+x_%7Bi%7D%2B%5Cfrac%7Bb%7D%7B%7C%7Cw%7C%7C%7D%5Cright%29)

Similarly, the geometric margin between hyper-plane `(w,b) `and the training set *T* is the minimum of ![{\gamma}_{i}](https://www.zhihu.com/equation?tex=%7B%5Cgamma%7D_%7Bi%7D)

![{\gamma}=\min\limits_{i=1,...,N} {\gamma}_{i}](https://www.zhihu.com/equation?tex=%7B%5Cgamma%7D%3D%5Cmin%5Climits_%7Bi%3D1%2C...%2CN%7D+%7B%5Cgamma%7D_%7Bi%7D)

Now, we can conclude the relationship between functional margin and geometric margin:

![\gamma_{i}=\frac{\hat{\gamma_{i}}}{||w||}](https://www.zhihu.com/equation?tex=%5Cgamma_%7Bi%7D%3D%5Cfrac%7B%5Chat%7B%5Cgamma_%7Bi%7D%7D%7D%7B%7C%7Cw%7C%7C%7D)

![\gamma=\frac{\hat{\gamma}}{||w||}](https://www.zhihu.com/equation?tex=%5Cgamma%3D%5Cfrac%7B%5Chat%7B%5Cgamma%7D%7D%7B%7C%7Cw%7C%7C%7D)

## SVM Model

The SVM model consists of optimization problem, optimization algorithm and classify.

### Optimization Problem

When the dataset is **linearly separable**, the supports vectors are the samples which are nearest to the hyper-plane as shown below.

![Image 9](https://www.codeproject.com/KB/AI/4064358/39a5198c-f369-43fd-8146-51f4642857e6.Png)

The samples on `H1` and `H2` are the support vectors.The distance between `H1 `and` H2` is called **hard margin**. Then, the optimization problem of SVM is:

![\min\limits_{w,b}\ \frac{1}{2}||w||^{2}\\ s.t.\ y_{i}\left(w\cdot x+b\right)-1\geq 0,\ i=1,2,...,N](https://www.zhihu.com/equation?tex=%5Cmin%5Climits_%7Bw%2Cb%7D%5C+%5Cfrac%7B1%7D%7B2%7D%7C%7Cw%7C%7C%5E%7B2%7D%5C%5C+s.t.%5C+y_%7Bi%7D%5Cleft%28w%5Ccdot+x%2Bb%5Cright%29-1%5Cgeq+0%2C%5C+i%3D1%2C2%2C...%2CN)

When the dataset is **not linearly separable**, some samples in the training set don't satisfy the condition that the functional margin is larger than or equal to `1`. To solve the problem, we add a **slack variable** ![\xi_{i}\geq0](https://www.zhihu.com/equation?tex=%5Cxi_%7Bi%7D%5Cgeq0) for each sample` (xi, yi)`. Then, the constraint is:

![y_{i}\left(w\cdot x+b\right)\geq 1-\xi_{i}](https://www.zhihu.com/equation?tex=y_%7Bi%7D%5Cleft%28w%5Ccdot+x%2Bb%5Cright%29%5Cgeq+1-%5Cxi_%7Bi%7D)

Meanwhile, add a cost for each slack variable. The target function is:

![\frac{1}{2}||w||^{2}+C\sum_{i=1}^{N}\xi_{i}](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B2%7D%7C%7Cw%7C%7C%5E%7B2%7D%2BC%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cxi_%7Bi%7D)

where `C `is the **penalty coefficient**. When `C` is large, the punishment of misclassification will be increased,  whereas the punishment of misclassification will be reduced. Then, the  optimization problem of SVM is:

![\min\limits_{w,b}\ \frac{1}{2}||w||^{2}+C\sum_{i=1}^{N}\xi_{i}\\ s.t.\ y_{i}\left(w\cdot x+b\right)\geq 1-\xi_{i},\ i=1,2,...,N\\ \xi_{i}\geq0,\ i=1,2,...,N](https://www.zhihu.com/equation?tex=%5Cmin%5Climits_%7Bw%2Cb%7D%5C+%5Cfrac%7B1%7D%7B2%7D%7C%7Cw%7C%7C%5E%7B2%7D%2BC%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cxi_%7Bi%7D%5C%5C+s.t.%5C+y_%7Bi%7D%5Cleft%28w%5Ccdot+x%2Bb%5Cright%29%5Cgeq+1-%5Cxi_%7Bi%7D%2C%5C+i%3D1%2C2%2C...%2CN%5C%5C+%5Cxi_%7Bi%7D%5Cgeq0%2C%5C+i%3D1%2C2%2C...%2CN)

The support vectors are on the border of margin or between the border and the hyper-plane as shown below. In this case, the margin is called **soft margin**.

### ![Image 15](https://www.codeproject.com/KB/AI/4064358/1785c9fa-3b9f-461b-9ec4-4b2066699aac.Png)

It needs to apply **kernel trick** to transform the data from original space into feature space when the dataset is not linearly separable. The function for kernel trick is called **kernel function**, which is defined as:

![K\left(x,z\right)=\phi\left(x\right)\cdot \phi\left(z\right)](https://www.zhihu.com/equation?tex=K%5Cleft%28x%2Cz%5Cright%29%3D%5Cphi%5Cleft%28x%5Cright%29%5Ccdot+%5Cphi%5Cleft%28z%5Cright%29)

where ![\phi\left(x\right)](https://www.zhihu.com/equation?tex=%5Cphi%5Cleft%28x%5Cright%29) is a mapping from input space to feature space, namely,

![\phi\left(x\right):\chi\rightarrow\mathcal{H}](https://www.zhihu.com/equation?tex=%5Cphi%5Cleft%28x%5Cright%29%3A%5Cchi%5Crightarrow%5Cmathcal%7BH%7D)

The code of kernel trick is shown below:

Hide   Copy Code

```
def kernelTransformation(self, data, sample, kernel):
        sample_num, feature_dim = np.shape(data)
        K = np.zeros([sample_num])
        if kernel == "linear":  # linear function
            K = np.dot(data, sample.T)
        elif kernel == "poly":  # polynomial function
            K = (np.dot(data, sample.T) + self.c) ** self.n
        elif kernel == "sigmoid":  # sigmoid function
            K = np.tanh(self.g * np.dot(data, sample.T) + self.c)
        elif kernel == "rbf":  # Gaussian function
            for i in range(sample_num):
                delta = data[i, :] - sample
                K[i] = np.dot(delta, delta.T)
            K = np.exp(-self.g * K)
        else:
            raise NameError('Unrecognized kernel function')
        return K
```

After applying kernel trick, the optimization problem of SVM is:

![\min\limits_{\alpha}\ \ \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_{i}\alpha_{j}y_{i}y_{j}K\left(x_{i},y_{j}\right)-\sum_{i=1}^{N}\alpha_{i}\\s.t.\ \ \sum_{i=1}^{N}\alpha_{i}y_{i}=0\\ 0\leq\alpha_{i}\leq C,i=1,2,...,N](https://www.zhihu.com/equation?tex=%5Cmin%5Climits_%7B%5Calpha%7D%5C+%5C+%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Csum_%7Bj%3D1%7D%5E%7BN%7D%5Calpha_%7Bi%7D%5Calpha_%7Bj%7Dy_%7Bi%7Dy_%7Bj%7DK%5Cleft%28x_%7Bi%7D%2Cy_%7Bj%7D%5Cright%29-%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Calpha_%7Bi%7D%5C%5Cs.t.%5C+%5C+%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Calpha_%7Bi%7Dy_%7Bi%7D%3D0%5C%5C+0%5Cleq%5Calpha_%7Bi%7D%5Cleq+C%2Ci%3D1%2C2%2C...%2CN)

where ![\alpha_{i}](https://www.zhihu.com/equation?tex=%5Calpha_%7Bi%7D) is the Lagrange factor.

### Optimization Algorithm

It is feasible to solve the SVM optimization problem with traditional convex quadratic programming algorithms. However, when the training set is large, the algorithms will take a long time. To solve the problem,  Platt proposed an efficient algorithm called **Sequential Minimal Optimization** (SMO).

SMO is a kind of heuristic algorithm. When all the variables follow the **KKT condition**, the optimization problem is solved. Else, choose two variables and fix  other variables and construct a convex quadratic programming problem  with the two variables. The problem has two variables: one chooses the  alpha which violated KKT conditions, the other is determined by the  constraints. Denote, the ![\alpha_{1,}\alpha_{2}](https://www.zhihu.com/equation?tex=%5Calpha_%7B1%2C%7D%5Calpha_%7B2%7D) are the variables, fix ![\alpha_{3},\alpha_{4},...,\alpha_{N}](https://www.zhihu.com/equation?tex=%5Calpha_%7B3%7D%2C%5Calpha_%7B4%7D%2C...%2C%5Calpha_%7BN%7D). Thus, ![\alpha_{1}](https://www.zhihu.com/equation?tex=%5Calpha_%7B1%7D) is calculated by:

![\alpha_{1}=-y_{1}\sum_{i=2}^{N}\alpha_{i}y_{i}](https://www.zhihu.com/equation?tex=%5Calpha_%7B1%7D%3D-y_%7B1%7D%5Csum_%7Bi%3D2%7D%5E%7BN%7D%5Calpha_%7Bi%7Dy_%7Bi%7D)

If ![\alpha_{2}](https://www.zhihu.com/equation?tex=%5Calpha_%7B2%7D) is determined , ![\alpha_{1}](https://www.zhihu.com/equation?tex=%5Calpha_%7B1%7D) is determined accordingly. In each iteration of SMO, two variables are  updated till the problem solved. Then, the optimization problem of SVM  is:

![\min\limits_{\alpha_{1},\alpha_{2}} W\left(\alpha_{1},\alpha_{2}\right)=\frac{1}{2}K_{11}\alpha_{1}^{2}+\frac{1}{2}K_{22}\alpha_{2}^{2}+y_{1}y_{2}K_{12}\alpha_{1}\alpha_{2}-\\\left(\alpha_{1}+\alpha_{2}\right)+y_{1}\alpha_{1}\sum_{i=3}^{N}y_{i}\alpha_{i}K_{i1}+y_{2}\alpha_{2}\sum_{i=3}^{N}y_{i}\alpha_{i}K_{i2}\\s.t.\  \ \ \alpha_{1}y_{1}+\alpha_{2}y_{2}=-\sum_{i=3}^{N}y_{i}\alpha_{i}\\ 0\leq\alpha_{i}\leq C,i=1,2](https://www.zhihu.com/equation?tex=%5Cmin%5Climits_%7B%5Calpha_%7B1%7D%2C%5Calpha_%7B2%7D%7D+W%5Cleft%28%5Calpha_%7B1%7D%2C%5Calpha_%7B2%7D%5Cright%29%3D%5Cfrac%7B1%7D%7B2%7DK_%7B11%7D%5Calpha_%7B1%7D%5E%7B2%7D%2B%5Cfrac%7B1%7D%7B2%7DK_%7B22%7D%5Calpha_%7B2%7D%5E%7B2%7D%2By_%7B1%7Dy_%7B2%7DK_%7B12%7D%5Calpha_%7B1%7D%5Calpha_%7B2%7D-%5C%5C%5Cleft%28%5Calpha_%7B1%7D%2B%5Calpha_%7B2%7D%5Cright%29%2By_%7B1%7D%5Calpha_%7B1%7D%5Csum_%7Bi%3D3%7D%5E%7BN%7Dy_%7Bi%7D%5Calpha_%7Bi%7DK_%7Bi1%7D%2By_%7B2%7D%5Calpha_%7B2%7D%5Csum_%7Bi%3D3%7D%5E%7BN%7Dy_%7Bi%7D%5Calpha_%7Bi%7DK_%7Bi2%7D%5C%5Cs.t.%5C++%5C+%5C+%5Calpha_%7B1%7Dy_%7B1%7D%2B%5Calpha_%7B2%7Dy_%7B2%7D%3D-%5Csum_%7Bi%3D3%7D%5E%7BN%7Dy_%7Bi%7D%5Calpha_%7Bi%7D%5C%5C+0%5Cleq%5Calpha_%7Bi%7D%5Cleq+C%2Ci%3D1%2C2)

When there is only two variable, it is a simple quadratic programming problem, as shown below:

Because the constraint is:

![\alpha_{1}y_{1}+\alpha_{2}y_{2}=-\sum_{i=3}^{N}y_{i}\alpha_{i} = k](https://www.zhihu.com/equation?tex=%5Calpha_%7B1%7Dy_%7B1%7D%2B%5Calpha_%7B2%7Dy_%7B2%7D%3D-%5Csum_%7Bi%3D3%7D%5E%7BN%7Dy_%7Bi%7D%5Calpha_%7Bi%7D+%3D+k)

when ![y_{1}=y_{2}](https://www.zhihu.com/equation?tex=y_%7B1%7D%3Dy_%7B2%7D), ![\alpha_{1}+\alpha_{2}=k](https://www.zhihu.com/equation?tex=%5Calpha_%7B1%7D%2B%5Calpha_%7B2%7D%3Dk)

when ![y_{1}\ne y_{2}](https://www.zhihu.com/equation?tex=y_%7B1%7D%5Cne+y_%7B2%7D), because ![y_{1},y_{2}\in\left\{1，-1\right\}](https://www.zhihu.com/equation?tex=y_%7B1%7D%2Cy_%7B2%7D%5Cin%5Cleft%5C%7B1%EF%BC%8C-1%5Cright%5C%7D),![\alpha_{1}-\alpha_{2}=k](https://www.zhihu.com/equation?tex=%5Calpha_%7B1%7D-%5Calpha_%7B2%7D%3Dk).

![Image 34](https://www.codeproject.com/KB/AI/4064358/239a2866-2951-48c1-be35-e535054422de.Png)

The optimized value ![\alpha_{2}^{new}](https://www.zhihu.com/equation?tex=%5Calpha_%7B2%7D%5E%7Bnew%7D) follows:

![L\leq\alpha_{2}^{new}\leq H](https://www.zhihu.com/equation?tex=L%5Cleq%5Calpha_%7B2%7D%5E%7Bnew%7D%5Cleq+H)

where `L `and `H` are the lower bound and upper bound of the diagonal line, which is calculated by:

![L,H=\left\{ \begin{aligned} L= max\left(0, \alpha_{2}^{old}-\alpha_{1}^{old}\right),H= min\left(C,C+ \alpha_{2}^{old}-\alpha_{1}^{old}\right) if\ y_{1}\ne y_{2} \\ L= max\left(0, \alpha_{2}^{old}+\alpha_{1}^{old}-C\right),H= min\left(C,\alpha_{2}^{old}+\alpha_{1}^{old}\right) if\ y_{1}= y_{2}  \\  \end{aligned} \right.](https://www.zhihu.com/equation?tex=L%2CH%3D%5Cleft%5C%7B+%5Cbegin%7Baligned%7D+L%3D+max%5Cleft%280%2C+%5Calpha_%7B2%7D%5E%7Bold%7D-%5Calpha_%7B1%7D%5E%7Bold%7D%5Cright%29%2CH%3D+min%5Cleft%28C%2CC%2B+%5Calpha_%7B2%7D%5E%7Bold%7D-%5Calpha_%7B1%7D%5E%7Bold%7D%5Cright%29+if%5C+y_%7B1%7D%5Cne+y_%7B2%7D+%5C%5C+L%3D+max%5Cleft%280%2C+%5Calpha_%7B2%7D%5E%7Bold%7D%2B%5Calpha_%7B1%7D%5E%7Bold%7D-C%5Cright%29%2CH%3D+min%5Cleft%28C%2C%5Calpha_%7B2%7D%5E%7Bold%7D%2B%5Calpha_%7B1%7D%5E%7Bold%7D%5Cright%29+if%5C+y_%7B1%7D%3D+y_%7B2%7D++%5C%5C++%5Cend%7Baligned%7D+%5Cright.+)

The uncutting optimized value ![\alpha_{2}^{new,unc}](https://www.zhihu.com/equation?tex=%5Calpha_%7B2%7D%5E%7Bnew%2Cunc%7D) follows:

![\alpha_{2}^{new,unc}=\alpha_{2}^{old}+\frac{y_{2}\left(E_{1}-E{2}\right)}{\eta}](https://www.zhihu.com/equation?tex=%5Calpha_%7B2%7D%5E%7Bnew%2Cunc%7D%3D%5Calpha_%7B2%7D%5E%7Bold%7D%2B%5Cfrac%7By_%7B2%7D%5Cleft%28E_%7B1%7D-E%7B2%7D%5Cright%29%7D%7B%5Ceta%7D)

where E1 and E2  are the difference between the prediction value `g(x)` and the real value. `g(x)` is defined as:

![g\left(x\right)=\sum_{i=1}^{N}\alpha_{i}y_{i}K\left(x_{i},x\right)+b](https://www.zhihu.com/equation?tex=g%5Cleft%28x%5Cright%29%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Calpha_%7Bi%7Dy_%7Bi%7DK%5Cleft%28x_%7Bi%7D%2Cx%5Cright%29%2Bb)

![\eta=K_{11}+K_{22}-2K_{12}](https://www.zhihu.com/equation?tex=%5Ceta%3DK_%7B11%7D%2BK_%7B22%7D-2K_%7B12%7D)

So far, we get feasible solutions for ![\alpha_{1,}\alpha_{2}](https://www.zhihu.com/equation?tex=%5Calpha_%7B1%2C%7D%5Calpha_%7B2%7D):

![\alpha_{2}^{new}=\left\{\begin{aligned} &H,&\alpha_{2}^{new,unc}>H\\ &\alpha_{2}^{new,unc},&L\leq\alpha_{2}^{new,unc}\leq H\\ &L,&\alpha_{2}^{new,unc}<L\end{aligned}\right.](https://www.zhihu.com/equation?tex=%5Calpha_%7B2%7D%5E%7Bnew%7D%3D%5Cleft%5C%7B%5Cbegin%7Baligned%7D+%26H%2C%26%5Calpha_%7B2%7D%5E%7Bnew%2Cunc%7D%3EH%5C%5C+%26%5Calpha_%7B2%7D%5E%7Bnew%2Cunc%7D%2C%26L%5Cleq%5Calpha_%7B2%7D%5E%7Bnew%2Cunc%7D%5Cleq+H%5C%5C+%26L%2C%26%5Calpha_%7B2%7D%5E%7Bnew%2Cunc%7D%3CL%5Cend%7Baligned%7D%5Cright.)

![\alpha_{1}^{new}=\alpha_{1}^{old}+y_{1}y_{2}\left(\alpha_{2}^{old}-\alpha_{2}^{new}\right)](https://www.zhihu.com/equation?tex=%5Calpha_%7B1%7D%5E%7Bnew%7D%3D%5Calpha_%7B1%7D%5E%7Bold%7D%2By_%7B1%7Dy_%7B2%7D%5Cleft%28%5Calpha_%7B2%7D%5E%7Bold%7D-%5Calpha_%7B2%7D%5E%7Bnew%7D%5Cright%29)

There are two loops in SMO, namely, outside loop and inner loop.

In the **outside loop**, choose the alpha which violated KKT conditions, namely,

![\alpha_{i}=0\Leftrightarrow y_{i}g\left(x_{i}\right)\geq1\\ 0<\alpha_{i}<C\Leftrightarrow y_{i}g\left(x_{i}\right)=1\\ \alpha_{i}=C\Leftrightarrow y_{i}g\left(x_{i}\right)\leq1](https://www.zhihu.com/equation?tex=%5Calpha_%7Bi%7D%3D0%5CLeftrightarrow+y_%7Bi%7Dg%5Cleft%28x_%7Bi%7D%5Cright%29%5Cgeq1%5C%5C+0%3C%5Calpha_%7Bi%7D%3CC%5CLeftrightarrow+y_%7Bi%7Dg%5Cleft%28x_%7Bi%7D%5Cright%29%3D1%5C%5C+%5Calpha_%7Bi%7D%3DC%5CLeftrightarrow+y_%7Bi%7Dg%5Cleft%28x_%7Bi%7D%5Cright%29%5Cleq1)

First, search the samples follow ![0<\alpha_{i}<C](https://www.zhihu.com/equation?tex=0%3C%5Calpha_%7Bi%7D%3CC).If all the samples follow the condition, search the whole dataset.

In the **inner loop**, first search the ![\alpha_{2}](https://www.zhihu.com/equation?tex=%5Calpha_%7B2%7D) which make ![\left|E_{1}-E_{2}\right|](https://www.zhihu.com/equation?tex=%5Cleft%7CE_%7B1%7D-E_%7B2%7D%5Cright%7C) maximum. If the chosen ![\alpha_{2}](https://www.zhihu.com/equation?tex=%5Calpha_%7B2%7D) doesn't decrease enough, first search the samples on the margin border. If the chosen ![\alpha_{2}](https://www.zhihu.com/equation?tex=%5Calpha_%7B2%7D) decreases enough, stop search. Else, search the whole dataset. If we can find a feasible ![\alpha_{2}](https://www.zhihu.com/equation?tex=%5Calpha_%7B2%7D) after searching the whole dataset, we will choose a new ![\alpha_{1}](https://www.zhihu.com/equation?tex=%5Calpha_%7B1%7D).

In each iteration, we updated `b `by:

![b_{1}^{new}=-E_{1}-y_{1}K_{11}\left(\alpha_{1}^{new}-\alpha_{1}^{old}\right)-y_{2}K_{21}\left(\alpha_{2}^{new}-\alpha_{2}^{old}\right)+b^{old}](https://www.zhihu.com/equation?tex=b_%7B1%7D%5E%7Bnew%7D%3D-E_%7B1%7D-y_%7B1%7DK_%7B11%7D%5Cleft%28%5Calpha_%7B1%7D%5E%7Bnew%7D-%5Calpha_%7B1%7D%5E%7Bold%7D%5Cright%29-y_%7B2%7DK_%7B21%7D%5Cleft%28%5Calpha_%7B2%7D%5E%7Bnew%7D-%5Calpha_%7B2%7D%5E%7Bold%7D%5Cright%29%2Bb%5E%7Bold%7D)

![b_{2}^{new}=-E_{2}-y_{1}K_{12}\left(\alpha_{1}^{new}-\alpha_{1}^{old}\right)-y_{2}K_{22}\left(\alpha_{2}^{new}-\alpha_{2}^{old}\right)+b^{old}](https://www.zhihu.com/equation?tex=b_%7B2%7D%5E%7Bnew%7D%3D-E_%7B2%7D-y_%7B1%7DK_%7B12%7D%5Cleft%28%5Calpha_%7B1%7D%5E%7Bnew%7D-%5Calpha_%7B1%7D%5E%7Bold%7D%5Cright%29-y_%7B2%7DK_%7B22%7D%5Cleft%28%5Calpha_%7B2%7D%5E%7Bnew%7D-%5Calpha_%7B2%7D%5E%7Bold%7D%5Cright%29%2Bb%5E%7Bold%7D)

![b=\left\{\begin{aligned} &b_{1}^{new},&if\ 0<\alpha_{1}^{new}\\ &b_{2}^{new},&if\ 0<\alpha_{2}^{new}\\ &\frac{b_{1}^{new}+b_{2}^{new}}{2},&else\end{aligned}\right.](https://www.zhihu.com/equation?tex=b%3D%5Cleft%5C%7B%5Cbegin%7Baligned%7D+%26b_%7B1%7D%5E%7Bnew%7D%2C%26if%5C+0%3C%5Calpha_%7B1%7D%5E%7Bnew%7D%5C%5C+%26b_%7B2%7D%5E%7Bnew%7D%2C%26if%5C+0%3C%5Calpha_%7B2%7D%5E%7Bnew%7D%5C%5C+%26%5Cfrac%7Bb_%7B1%7D%5E%7Bnew%7D%2Bb_%7B2%7D%5E%7Bnew%7D%7D%7B2%7D%2C%26else%5Cend%7Baligned%7D%5Cright.)

For convenience, we have to store the value of ![E_{i}](https://www.zhihu.com/equation?tex=E_%7Bi%7D), which is calculated by:

![E_{i}^{new}=\sum_{s}y_{j}\alpha_{j}K\left(x_{i},y_{j}\right)+b^{new}-y_{i}](https://www.zhihu.com/equation?tex=E_%7Bi%7D%5E%7Bnew%7D%3D%5Csum_%7Bs%7Dy_%7Bj%7D%5Calpha_%7Bj%7DK%5Cleft%28x_%7Bi%7D%2Cy_%7Bj%7D%5Cright%29%2Bb%5E%7Bnew%7D-y_%7Bi%7D)

The search and update code is shown below:

Hide   Shrink ![img](https://www.codeproject.com/images/arrow-up-16.png)   Copy Code

```
def innerLoop(self, i):
        Ei = self.calculateErrors(i)
        if self.checkKKT(i):

            j, Ej = self.selectAplha2(i, Ei)          # select alpha2 according to alpha1

            # copy alpha1 and alpha2
            old_alpha1 = self.alphas[i]
            old_alpha2 = self.alphas[j]

            # determine the range of alpha2 L and H      in page of 126
            # if y1 != y2    L = max(0, old_alpha2 - old_alpha1), 
                             H = min(C, C + old_alpha2 - old_alpha1)
            # if y1 == y2    L = max(0, old_alpha2 + old_alpha1 - C), 
                             H = min(C, old_alpha2 + old_alpha1)
            if self.train_label[i] != self.train_label[j]:
                L = max(0, old_alpha2 - old_alpha1)
                H = min(self.C, self.C + old_alpha2 - old_alpha1)
            else:
                L = max(0, old_alpha2 + old_alpha1 - self.C)
                H = min(self.C, old_alpha2 + old_alpha2)

            if L == H:
                # print("L == H")
                return 0

            # calculate eta in page of 127 Eq.(7.107)
            # eta = K11 + K22 - 2K12
            K11 = self.K[i, i]
            K12 = self.K[i, j]
            K21 = self.K[j, i]
            K22 = self.K[j, j]
            eta = K11 + K22 - 2 * K12
            if eta <= 0:
                # print("eta <= 0")
                return 0

            # update alpha2 and its error in page of 127 Eq.(7.106) and Eq.(7.108)
            self.alphas[j] = old_alpha2 + self.train_label[j]*(Ei - Ej)/eta
            self.alphas[j] = self.updateAlpha2(self.alphas[j], L, H)
            new_alphas2 = self.alphas[j]
            self.upadateError(j)


            # update the alpha1 and its error in page of 127 Eq.(7.109)
            # new_alpha1 = old_alpha1 + y1y2(old_alpha2 - new_alpha2)
            new_alphas1 = old_alpha1 + self.train_label[i] * 
                          self.train_label[j] * (old_alpha2 - new_alphas2)
            self.alphas[i] = new_alphas1
            self.upadateError(i)

            # determine b in page of 130 Eq.(7.115) and Eq.(7.116)
            # new_b1 = -E1 - y1K11(new_alpha1 - old_alpha1) - 
                             y2K21(new_alpha2 - old_alpha2) + old_b
            # new_b2 = -E2 - y1K12(new_alpha1 - old_alpha1) - 
                             y2K22(new_alpha2 - old_alpha2) + old_b
            b1 = - Ei - self.train_label[i] * K11 * (old_alpha1 - self.alphas[i]) - 
                        self.train_label[j] * K21 * (old_alpha2 - self.alphas[j]) + self.b
            b2 = - Ej - self.train_label[i] * K12 * (old_alpha1 - self.alphas[i]) - 
                        self.train_label[j] * K22 * (old_alpha2 - self.alphas[j]) + self.b
            if (self.alphas[i] > 0) and (self.alphas[i] < self.C):
                self.b = b1
            elif (self.alphas[j] > 0) and (self.alphas[j] < self.C):
                self.b = b2
            else:
                self.b = (b1 + b2)/2.0

            return 1
        else:
            return 0
```

### Classify

We can make a prediction using the optimized parameters, which is given by:

![f\left(x\right)=sign\left(\sum_{i=1}^{N}\alpha_{i}^{*}y_{i}K\left(x,x_{i}\right)+b^{*}\right)](https://www.zhihu.com/equation?tex=f%5Cleft%28x%5Cright%29%3Dsign%5Cleft%28%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Calpha_%7Bi%7D%5E%7B%2A%7Dy_%7Bi%7DK%5Cleft%28x%2Cx_%7Bi%7D%5Cright%29%2Bb%5E%7B%2A%7D%5Cright%29)

Hide   Copy Code

```
for i in range(test_num):
    kernel_data = self.kernelTransformation(support_vectors, test_data[i, :], self.kernel)
    probability[i] = np.dot(kernel_data.T, np.multiply
                     (support_vectors_label, support_vectors_alphas)) + self.b
    if probability[i] > 0:
        prediction[i] = 1
    else:
        prediction[i] = -1
```

## Conclusion and Analysis

SVM is a more complex algorithm than previous algorithms. In this  article, we simplify the search process to make it a bit more easy to  understand. Finally, let's compare our SVM with the SVM in Sklearn and  the detection performance is displayed below:

![Image 59](https://www.codeproject.com/KB/AI/4064358/89f9bd40-e4c0-478e-991a-07b06313df41.Png)

The detection performance is a little worse than the sklearn's , which is because the SMO in our SVM is simpler than sklearn's.

The related code and dataset in this article can be found in [MachineLearning](https://github.com/DandelionLau/MachineLearning).