# Step-by-Step Guide to Implement Machine Learning IX - Tree Regression

![img](https://www.codeproject.com/script/Membership/ProfileImages/{6098a88e-6a47-4a71-915c-4577a7b84ee9}.jpg)

[danc1elion](https://www.codeproject.com/script/Membership/View.aspx?mid=14354398)

|      | Rate this: 			 				 			                      				   	 	         ![img](https://codeproject.freetls.fastly.net/script/Ratings/Images/stars-fill-lg.png)![img](https://codeproject.freetls.fastly.net/script/Ratings/Images/stars-empty-lg.png) 		  	 	5.00  (1 vote) |
| ---- | ------------------------------------------------------------ |
|      |                                                              |

​                                        28 May 2019[CPOL](http://www.codeproject.com/info/cpol10.aspx)                                    

Easy to implement machine learning



This article is an entry in our [Machine Learning and Artificial Intelligence Challenge](https://www.codeproject.com/Competitions/1024/The-Machine-Learning-and-Artificial-Intelligence-C.aspx). Articles in this sub-section are not required to be full articles so care should be taken when voting.

## Introduction

In the real world, some relationships are not linear. Thus, it is not suitable to apply linear regression to analysis those problem. To solve the problem, we could employ tree regression. The main idea of tree  regression it to divide the problem into smaller subproblems. If the  subproblem is linear, we can combine all the models of subproblems to  get the regression model for the entire problem.

## Regression Model

Tree regression is similar to decision tree, which consists of feature selection, generation of regression tree and regression.

### Feature selection

In decision tree, we select features according to information gain.  However, for regression tree, the prediction value is continuous, which  means the regression label is nearly unique for each sample. Thus,  empirical entropy lack the ability of characterization. So, we utilize  square error as the criterion for feature selection, namely,

 ![\sum_{x_{i}\in R_{m}}\left(y_{i}-f\left(x_{i}\right)\right)^{2}](https://www.zhihu.com/equation?tex=%5Csum_%7Bx_%7Bi%7D%5Cin+R_%7Bm%7D%7D%5Cleft%28y_%7Bi%7D-f%5Cleft%28x_%7Bi%7D%5Cright%29%5Cright%29%5E%7B2%7D)

where ***Rm\*** are the spaces divided by regression tree, ***f\*(\*x\*)** is given by

![f\left(x\right)=\sum_{m=1}^{M}c_{m}I\left(x\in R_{m}\right)](https://www.zhihu.com/equation?tex=f%5Cleft%28x%5Cright%29%3D%5Csum_%7Bm%3D1%7D%5E%7BM%7Dc_%7Bm%7DI%5Cleft%28x%5Cin+R_%7Bm%7D%5Cright%29)

Thus, no matter what the features of sample are, the outputs in the same space are the same. The output of ***Rm\*** is the average of all the samples' regression label in the space, namely

![c_{m}=\arg \left(y_{i}|x_{i}\in R_{m}\right)](https://www.zhihu.com/equation?tex=c_%7Bm%7D%3D%5Carg+%5Cleft%28y_%7Bi%7D%7Cx_%7Bi%7D%5Cin+R_%7Bm%7D%5Cright%29)

The feature selection for regression tree is similar to decision tree, which aim to minimum the loss function, namely,

![\min\limits_{j,s}\left[\min\limits_{c_{1}}\sum_{x_{i}\in R_{1}\left(j,s\right)}\left(y_{i}-c_{1}\right)+\min\limits_{c_{2}}\sum_{x_{i}\in R_{2}\left(j,s\right)}\left(y_{i}-c_{2}\right)\right]](https://www.zhihu.com/equation?tex=%5Cmin%5Climits_%7Bj%2Cs%7D%5Cleft%5B%5Cmin%5Climits_%7Bc_%7B1%7D%7D%5Csum_%7Bx_%7Bi%7D%5Cin+R_%7B1%7D%5Cleft%28j%2Cs%5Cright%29%7D%5Cleft%28y_%7Bi%7D-c_%7B1%7D%5Cright%29%2B%5Cmin%5Climits_%7Bc_%7B2%7D%7D%5Csum_%7Bx_%7Bi%7D%5Cin+R_%7B2%7D%5Cleft%28j%2Cs%5Cright%29%7D%5Cleft%28y_%7Bi%7D-c_%7B2%7D%5Cright%29%5Cright%5D)

### Generation of Regression Tree

We first define a data structure to save the tree node

Hide   Copy Code

```
class RegressionNode():    
    def __init__(self, index=-1, value=None, result=None, right_tree=None, left_tree=None):
        self.index = index
        self.value = value
        self.result = result
        self.right_tree = right_tree
        self.left_tree = left_tree
```

Like decision tree, suppose that we have selected the best feature and its corresponding value **(\*j\*, \*s\*)**, then we spilt the data by

![R_{1}\left(j,s\right)=\left\{ x|x^{(j)}\leq s\right\} ](https://www.zhihu.com/equation?tex=R_%7B1%7D%5Cleft%28j%2Cs%5Cright%29%3D%5Cleft%5C%7B+x%7Cx%5E%7B%28j%29%7D%5Cleq+s%5Cright%5C%7D+)

![R_{2}\left(j,s\right)=\left\{ x|x^{(j)}> s\right\}](https://www.zhihu.com/equation?tex=R_%7B2%7D%5Cleft%28j%2Cs%5Cright%29%3D%5Cleft%5C%7B+x%7Cx%5E%7B%28j%29%7D%3E+s%5Cright%5C%7D)

And the output of each binary is 

![c_{m}=\frac{1}{N_{m}}\sum_{x_{i}\in R_{m}\left(j,s\right)}y_{i},x\in R_{m}，m=1,2](https://www.zhihu.com/equation?tex=c_%7Bm%7D%3D%5Cfrac%7B1%7D%7BN_%7Bm%7D%7D%5Csum_%7Bx_%7Bi%7D%5Cin+R_%7Bm%7D%5Cleft%28j%2Cs%5Cright%29%7Dy_%7Bi%7D%2Cx%5Cin+R_%7Bm%7D%EF%BC%8Cm%3D1%2C2)

The generation of regression tree is nearly the same as that of decision tree, which will not be described here. You can read [Step-by-Step Guide To Implement Machine Learning II - Decision Tree](https://www.codeproject.com/Articles/4047359/Step-by-Step-Guide-To-Implement-Machine-Learning-2) to get more detail. If you still have question, please contact with me. I  am pleasure to help you solve any problem about regression tree.

Hide   Shrink ![img](https://www.codeproject.com/images/arrow-up-16.png)   Copy Code

```
def createRegressionTree(self, data):
    # if there is no feature
    if len(data) == 0:
        self.tree_node = treeNode(result=self.getMean(data[:, -1]))
        return self.tree_node

    sample_num, feature_dim = np.shape(data)

    best_criteria = None
    best_error = np.inf
    best_set = None
    initial_error = self.getVariance(data)

    # get the best split feature and value
    for index in range(feature_dim - 1):
        uniques = np.unique(data[:, index])
        for value in uniques:
            left_set, right_set = self.divideData(data, index, value)
            if len(left_set) < self.N or len(right_set) < self.N:
                continue
            new_error = self.getVariance(left_set) + self.getVariance(right_set)
            if new_error < best_error:
                best_criteria = (index, value)
                best_error = new_error
                best_set = (left_set, right_set)

    if best_set is None:
        self.tree_node = treeNode(result=self.getMean(data[:, -1]))
        return self.tree_node
    # if the descent of error is small enough, return the mean of the data
    elif abs(initial_error - best_error) < self.error_threshold:
        self.tree_node = treeNode(result=self.getMean(data[:, -1]))
        return self.tree_node
    # if the split data is small enough, return the mean of the data
    elif len(best_set[0]) < self.N or len(best_set[1]) < self.N:
        self.tree_node = treeNode(result=self.getMean(data[:, -1]))
        return self.tree_node
    else:
        ltree = self.createRegressionTree(best_set[0])
        rtree = self.createRegressionTree(best_set[1])
        self.tree_node = treeNode(index=best_criteria[0], value=best_criteria[1], left_tree=ltree, right_tree=rtree)
        return self.tree_node
```

### Regression

The principle of regression is like the binary sort tree, namely, **comparing the feature value stored in node with the corresponding feature value  of the test object. Then, turn to left subtree or right subtree  recursively** as shown below:

Hide   Copy Code

```
def classify(self, sample, tree):
    if tree.result is not None:
        return tree.result
    else:
        value = sample[tree.index]
        if value >= tree.value:
            branch = tree.right_tree
        else:
            branch = tree.left_tree
        return self.classify(sample, branch)
```

## Conclusion and Analysis

Classification tree and regression tree can be combined as  Classification and regression tree(CART). Indeed, there exists prune  process when generating tree of after generating tree. We skip them  because it is a little complex and not always effective. Finally, let's  compare our regression tree with the tree in Sklearn and the detection  performance is displayed below:

Sklearn tree regression performace:

![Image 8](https://www.codeproject.com/KB/AI/5061172/19d47e06-07ce-4535-8b4d-38844af2757f.Png)

Our tree regression performace:

![Image 9](https://www.codeproject.com/KB/AI/5061172/69810e82-8fd6-45b2-b32b-be0fa2913da3.Png)

 

![Image 10](https://www.codeproject.com/KB/AI/5061172/7cfd6f92-0cde-48aa-ba2e-0f36d745940a.Png)

Our tree regression takes a bit longer time than sklearn's.

The related code and dataset in this article can be found in [MachineLearning](https://github.com/DandelionLau/MachineLearning).