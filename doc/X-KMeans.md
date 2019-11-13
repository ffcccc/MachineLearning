# Step-by-Step Guide to Implement Machine Learning X - KMeans

![img](https://www.codeproject.com/script/Membership/ProfileImages/{6098a88e-6a47-4a71-915c-4577a7b84ee9}.jpg)

[danc1elion](https://www.codeproject.com/script/Membership/View.aspx?mid=14354398)

|      | Rate this: 			 				 			                      				   	 	         ![img](https://codeproject.freetls.fastly.net/script/Ratings/Images/stars-fill-lg.png)![img](https://codeproject.freetls.fastly.net/script/Ratings/Images/stars-empty-lg.png) 		  	 	4.33  (5 votes) |
| ---- | ------------------------------------------------------------ |
|      |                                                              |

​                                        3 Jun 2019[CPOL](http://www.codeproject.com/info/cpol10.aspx)                                    

Easy to implement machine learning



This article is an entry in our [Machine Learning and Artificial Intelligence Challenge](https://www.codeproject.com/Competitions/1024/The-Machine-Learning-and-Artificial-Intelligence-C.aspx). Articles in this sub-section are not required to be full articles so care should be taken when voting.

## Introduction

KMeans is a simple clustering algorithm, which calculates the  distance between samples and centroid to generate clusters. K is the  number of clusters given by user. At the initial time, the K clusters  are chosen randomly, they are adjusted at each iteration to get the  optimal results. The centroid is the mean of samples in its  corresponding cluster, thus, the algorithm is called K"Means".

## KMeans Model

### KMeans

The normal KMeans algorithm is really simple. Denote the K clusters as ![\mu_{1},\mu_{2},...\mu_{k}](https://www.zhihu.com/equation?tex=%5Cmu_%7B1%7D%2C%5Cmu_%7B2%7D%2C...%5Cmu_%7Bk%7D), and the number of samples in each cluster as ![N_{1},N_{2},...,N_{k}](https://www.zhihu.com/equation?tex=N_%7B1%7D%2CN_%7B2%7D%2C...%2CN_%7Bk%7D). The loss function of KMeans is:

![J\left(\mu_{1},\mu_{2},...\mu_{k}\right)=\frac{1}{2}\sum_{j=1}^{K}\sum_{i=1}^{N_{j}}\left(x_{i}-\mu_{j}\right)^{2}](https://www.zhihu.com/equation?tex=J%5Cleft%28%5Cmu_%7B1%7D%2C%5Cmu_%7B2%7D%2C...%5Cmu_%7Bk%7D%5Cright%29%3D%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bj%3D1%7D%5E%7BK%7D%5Csum_%7Bi%3D1%7D%5E%7BN_%7Bj%7D%7D%5Cleft%28x_%7Bi%7D-%5Cmu_%7Bj%7D%5Cright%29%5E%7B2%7D).

Calculate the derivative of loss function:

![\frac{\partial J}{\partial\mu_{j}}=-\sum_{i=1}^{N_{j}}\left(x_{i}-\mu_{j}\right)](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+J%7D%7B%5Cpartial%5Cmu_%7Bj%7D%7D%3D-%5Csum_%7Bi%3D1%7D%5E%7BN_%7Bj%7D%7D%5Cleft%28x_%7Bi%7D-%5Cmu_%7Bj%7D%5Cright%29)

Then, make the derivative equal to 0, we can obtain:

![\mu_{j}=\frac{1}{N_{j}}\sum_{i=1}^{N_{j}}x_{i}](https://www.zhihu.com/equation?tex=%5Cmu_%7Bj%7D%3D%5Cfrac%7B1%7D%7BN_%7Bj%7D%7D%5Csum_%7Bi%3D1%7D%5E%7BN_%7Bj%7D%7Dx_%7Bi%7D)

namely, the centroid is the means of samples in its corresponding cluster. The code of KMeans is shown below:

Hide   Copy Code

```
def kmeans(self, train_data, k):
    sample_num = len(train_data)
    distances = np.zeros([sample_num, 2])                      # (index, distance)
    centers = self.createCenter(train_data)
    centers, distances = self.adjustCluster(centers, distances, train_data, self.k)
    return centers, distances
```

where `adjustCluster()` is to adjust centroid after determining the initial centroids, which aims to minimize the loss function. The code of `adjustCluster` is:

Hide   Copy Code

```
def adjustCluster(self, centers, distances, train_data, k):
    sample_num = len(train_data)
    flag = True  # If True, update cluster_center
    while flag:
        flag = False
        d = np.zeros([sample_num, len(centers)])
        for i in range(len(centers)):
            # calculate the distance between each sample and each cluster center
            d[:, i] = self.calculateDistance(train_data, centers[i])

        # find the minimum distance between each sample and each cluster center
        old_label = distances[:, 0].copy()
        distances[:, 0] = np.argmin(d, axis=1)
        distances[:, 1] = np.min(d, axis=1)
        if np.sum(old_label - distances[:, 0]) != 0:
            flag = True
            # update cluster_center by calculating the mean of each cluster
            for j in range(k):
                current_cluster =
                      train_data[distances[:, 0] == j]  # find the samples belong
                                                        # to the j-th cluster center
                if len(current_cluster) != 0:
                    centers[j, :] = np.mean(current_cluster, axis=0)
    return centers, distances
```

### Bisecting KMeans

Because KMeans may get a local optimation result, to solve the  problem, we introduce another KMeans algorithm called bisecting KMeans.  In bisecting KMeans, all the samples are regarded as a cluster at  initial, and the cluster is divided into two parts. Then, choose one  part of the divided cluster to bisect again and again till the number of cluster is K. We bisect the cluster according to minimum **Sum of Squared Error**(SSE). Denote the current ***n*** clusters as:

![C=\left\{c_{1},c_{2},...,c_{n}\right\},n<k](https://www.zhihu.com/equation?tex=C%3D%5Cleft%5C%7Bc_%7B1%7D%2Cc_%7B2%7D%2C...%2Cc_%7Bn%7D%5Cright%5C%7D%2Cn%3Ck)

We choose a cluster ![c_{i}](https://www.zhihu.com/equation?tex=c_%7Bi%7D) in ![C](https://www.zhihu.com/equation?tex=C) and bisect it into two parts ![c_{i1},c_{i2}](https://www.zhihu.com/equation?tex=c_%7Bi1%7D%2Cc_%7Bi2%7D) using normal KMeans. The SSE is:

![SSE_{i}=SSE\left(c_{i1},c_{i2}\right)+SSE\left(C-c_{i}\right)](https://www.zhihu.com/equation?tex=SSE_%7Bi%7D%3DSSE%5Cleft%28c_%7Bi1%7D%2Cc_%7Bi2%7D%5Cright%29%2BSSE%5Cleft%28C-c_%7Bi%7D%5Cright%29)

We choose the ![c_{i}](https://www.zhihu.com/equation?tex=c_%7Bi%7D) which can get a minimum of SSE as the cluster to be bisected, namely:

![index =arg\min SSE_{i}](https://www.zhihu.com/equation?tex=index+%3Darg%5Cmin+SSE_%7Bi%7D)

Repeat the above processes till the number of clusters is K.

Hide   Shrink ![img](https://www.codeproject.com/images/arrow-up-16.png)   Copy Code

```
def biKmeans(self, train_data):
    sample_num = len(train_data)
    distances = np.zeros([sample_num, 2])         # (index, distance)
    initial_center = np.mean(train_data, axis=0)  # initial cluster #shape (1, feature_dim)
    centers = [initial_center]                    # cluster list

    # clustering with the initial cluster center
    distances[:, 1] = np.power(self.calculateDistance(train_data, initial_center), 2)

    # generate cluster centers
    while len(centers) < self.k:
        # print(len(centers))
        min_SSE  = np.inf
        best_index = None
        best_centers = None
        best_distances = None

        # find the best split
        for j in range(len(centers)):
            centerj_data = train_data[distances[:, 0] == j]   # find the samples belonging
                                                              # to the j-th center
            split_centers, split_distances = self.kmeans(centerj_data, 2)
            split_SSE = np.sum(split_distances[:, 1]) ** 2    # calculate the distance
                                                              # for after clustering
            other_distances = distances[distances[:, 0] != j] # the samples don't belong
                                                              # to j-th center
            other_SSE = np.sum(other_distances[:, 1]) ** 2    # calculate the distance
                                                              # don't belong to j-th center

            # save the best split result
            if (split_SSE + other_SSE) < min_SSE:
                best_index = j
                best_centers = split_centers
                best_distances = split_distances
                min_SSE = split_SSE + other_SSE

        # save the spilt data
        best_distances[best_distances[:, 0] == 1, 0] = len(centers)
        best_distances[best_distances[:, 0] == 0, 0] = best_index

        centers[best_index] = best_centers[0, :]
        centers.append(best_centers[1, :])
        distances[distances[:, 0] == best_index, :] = best_distances
    centers = np.array(centers)   # transform form list to array
    return centers, distances
```

### KMeans++

Because the initial centroid has great effect on the performance of  KMeans, to solve the problem, we introduce another KMeans algorithm  called KMeans++. Denote the current ***n*** clusters as:

![C=\left\{c_{1},c_{2},...,c_{n}\right\},n<k](https://www.zhihu.com/equation?tex=C%3D%5Cleft%5C%7Bc_%7B1%7D%2Cc_%7B2%7D%2C...%2Cc_%7Bn%7D%5Cright%5C%7D%2Cn%3Ck)

When we choose the (n+1)-th centroid, the farther from the existing  centroids the sample is, the more probable it will be chosen as the new  centroid. First, we calculate the minimum distance between each sample  and the existing centroids:

![D\left(x_{i}\right)=\min\limits_{c_{j}\in C}\left(x_{i}-c_{j}\right)](https://www.zhihu.com/equation?tex=D%5Cleft%28x_%7Bi%7D%5Cright%29%3D%5Cmin%5Climits_%7Bc_%7Bj%7D%5Cin+C%7D%5Cleft%28x_%7Bi%7D-c_%7Bj%7D%5Cright%29)

Then, calculate the probability of each sample to be chosen as the next centroid:

![p_{i}=\frac{D\left(x_{i}\right)^{2}}{\sum_{x_{i}\in X}D\left(x_{i}\right)^{2}}](https://www.zhihu.com/equation?tex=p_%7Bi%7D%3D%5Cfrac%7BD%5Cleft%28x_%7Bi%7D%5Cright%29%5E%7B2%7D%7D%7B%5Csum_%7Bx_%7Bi%7D%5Cin+X%7DD%5Cleft%28x_%7Bi%7D%5Cright%29%5E%7B2%7D%7D)

Then, we use roulette wheel selection to get the next centroid. After determining K clusters, run `adjustCluster()` to adjust the result.

Hide   Shrink ![img](https://www.codeproject.com/images/arrow-up-16.png)   Copy Code

```
def kmeansplusplus(self,train_data):
    sample_num = len(train_data)
    distances = np.zeros([sample_num, 2])       # (index, distance)

    # randomly select a sample as the initial cluster
    initial_center = train_data[np.random.randint(0, sample_num-1)]
    centers = [initial_center]

    while len(centers) < self.k:
        d = np.zeros([sample_num, len(centers)])
        for i in range(len(centers)):
            # calculate the distance between each sample and each cluster center
            d[:, i] = self.calculateDistance(train_data, centers[i])

        # find the minimum distance between each sample and each cluster center
        distances[:, 0] = np.argmin(d, axis=1)
        distances[:, 1] = np.min(d, axis=1)

        # Roulette Wheel Selection
        prob = np.power(distances[:, 1], 2)/np.sum(np.power(distances[:, 1], 2))
        index = self.rouletteWheelSelection(prob, sample_num)
        new_center = train_data[index, :]
        centers.append(new_center)

    # adjust cluster
    centers = np.array(centers)   # transform form list to array
    centers, distances = self.adjustCluster(centers, distances, train_data, self.k)
    return centers, distances
```

## Conclusion and Analysis

Indeed, it is necessary to adjust clusters after determining how to  choose the parameter 'K', there exist some algorithms. Finally, let's  compare the performances of three kinds of clustering algorithms.

![Image 16](https://www.codeproject.com/KB/AI/5061517/f051ab13-4edd-40a9-81f6-9f8da6917b9e.Png)

![Image 17](https://www.codeproject.com/KB/AI/5061517/219afae3-d82d-4e81-9a0d-3db3c0b51f29.Png)

![Image 18](https://www.codeproject.com/KB/AI/5061517/7f2b58fc-90a0-4956-866c-1dba203637cd.Png)

![Image 19](https://www.codeproject.com/KB/AI/5061517/93b8e925-ea25-4a82-966d-483564d11f3f.Png)

It can be found that KMeans++ has the best performance.

The related code and dataset in this article can be found in [MachineLearning](https://github.com/DandelionLau/MachineLearning).