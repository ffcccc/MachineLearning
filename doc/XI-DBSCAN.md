# Step-by-Step Guide to Implement Machine Learning XI - DBSCAN

![img](https://www.codeproject.com/script/Membership/ProfileImages/{6098a88e-6a47-4a71-915c-4577a7b84ee9}.jpg)

[danc1elion](https://www.codeproject.com/script/Membership/View.aspx?mid=14354398)

|      | Rate this: 			 				 			                      				   	 	         ![img](https://codeproject.freetls.fastly.net/script/Ratings/Images/stars-fill-lg.png)![img](https://codeproject.freetls.fastly.net/script/Ratings/Images/stars-empty-lg.png) 		  	 	4.33  (2 votes) |
| ---- | ------------------------------------------------------------ |
|      |                                                              |

​                                        17 Jun 2019[CPOL](http://www.codeproject.com/info/cpol10.aspx)                                    

Easy to implement machine learning



This article is an entry in our [Machine Learning and Artificial Intelligence Challenge](https://www.codeproject.com/Competitions/1024/The-Machine-Learning-and-Artificial-Intelligence-C.aspx). Articles in this sub-section are not required to be full articles so care should be taken when voting.

## Introduction

Density-based spatial clustering of applications with noise(DBSCAN)  is a clustering algorithm based on density. Unlike other clustering  algorithms, DBSCAN regards the maximum set of density reachable samples  as the cluster. KMeans lacks the ability to cluster the nonspherical  data, while DBSCAN can cluster data with any shape.

## DBSCAN Model

### Preliminary

- **ε -neighbor**: Objects within a radius of ε from an object.
- **Core point**: One sample is a core point if it has more than a specified number of points (**m**) within ε- neighbor.
- **Directly density-reachable**: An object **q** is directly density-reachable from object **p** if **q** is within the ε-Neighborhood of **p** and **p** is a core object.
- **Density-reachable**: An object **q** is density-reachable from **p** if there is a chain of objects p1,…,pn, with p1=p, pn=q such that  pi+1is directly density-reachable from pi for all 1 <= i <= n (the difference between Directly density-reachable and Density-reachable is  that p is within the neighbor of q when Directly density-reachable while p is not within the neighbor of q when density-reachable)
- **Density-connectivity**: Object **p** is density-connected to object **q** if there is an object **o** such that both **p** and **q** are density-reachable from **o**
- **Noise**: objects which are not density-reachable from at least one core object

Let's give a example to explain the above terms as shown below, where **m** = 4. Because there are more than **m** samples in the ε- neighbors of all the red samples, they are all **core points** and **density-reachable**. B and C are not core points. The samples in the cyan circle are **directly density-reachable**. The samples in the brown circle are **density-connectivity**. N is not reachable to any other samples, thus it is a **noise point**.

![Image 1](https://www.codeproject.com/KB/AI/5129186/15fe9f12-cb98-4972-a5ae-c9b4382bca92.Png)

### Clustering Algorithm

First, we find all the core points by calculating the number of  samples within all the samples ε- neighbor. The remained samples are  marked noise point as temporarily. We provide several types of distance, namely:

- Euclidean distance: 

  ![L_{2}(x_{i},x_{j})=\left( \sum_{l=1}^{n}{|x_{i}^{(l)}-x_{j}^{(l)}|^{2}}\right)^\frac{1}{2}](https://www.zhihu.com/equation?tex=L_%7B2%7D%28x_%7Bi%7D%2Cx_%7Bj%7D%29%3D%5Cleft%28+%5Csum_%7Bl%3D1%7D%5E%7Bn%7D%7B%7Cx_%7Bi%7D%5E%7B%28l%29%7D-x_%7Bj%7D%5E%7B%28l%29%7D%7C%5E%7B2%7D%7D%5Cright%29%5E%5Cfrac%7B1%7D%7B2%7D)

- Cosine distance: 

  ![\cos\left(x_{i},x_{j}\right)=\frac{x_{i}\cdot x_{j} }{|| x_{i}||\ ||x_{j}||}](https://www.zhihu.com/equation?tex=%5Ccos%5Cleft(x_%7Bi%7D%2Cx_%7Bj%7D%5Cright)%3D%5Cfrac%7Bx_%7Bi%7D%5Ccdot%20x_%7Bj%7D%20%7D%7B%7C%7C%20x_%7Bi%7D%7C%7C%5C%20%7C%7Cx_%7Bj%7D%7C%7C%7D)

- Manhattan distance: 

  ![Mabhatten\left(x_{i},x_{j}\right)=\sum^{n}_{l=1}\left|x_{i}^{(l)}-x_{j}^{(l)}\right|](https://www.zhihu.com/equation?tex=Mabhatten%5Cleft(x_%7Bi%7D%2Cx_%7Bj%7D%5Cright)%3D%5Csum%5E%7Bn%7D_%7Bl%3D1%7D%5Cleft%7Cx_%7Bi%7D%5E%7B(l)%7D-x_%7Bj%7D%5E%7B(l)%7D%5Cright%7C)

Hide   Copy Code

```
def calculateDistance(self, x1, x2):
    if self.distance_type == "Euclidean":
        d = np.sqrt(np.sum(np.power(x1 - x2, 2), axis=1))
        #d = np.sqrt(np.sum(np.power(x1 - x2, 2)))
    elif self.distance_type == "Cosine":
        d = np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))
    elif self.distance_type == "Manhattan":
        d = np.sum(x1 - x2)
    else:
        print("Error Type!")
        sys.exit()
    return d
```

For a given sample, if the distance between this sample and another sample is less than ε, they are in the same neighbor.

Hide   Copy Code

```
def getCenters(self, train_data):
    neighbor = {}
    for i in range(len(train_data)):
        distance = self.calculateDistance(train_data[i], train_data)
        index = np.where(distance <= self.eps)[0]
        if len(index) > self.m:
            neighbor[i] = index
    return neighbor
```

Then, we choose an unvisited core point P randomly to merge the  samples which are density-connectivity with it. Next, we visit the  samples Q within ε- neighbor of P. If Q is a core point, Q is in the  same cluster of P and do the same process on Q (like depth first  search); else, visit the next sample. Specifically, the search process  in P's ε- neighbor is shown below:

- Choose core point A and find all the samples in its ε- neighbor.  The samples within A's ε- neighbor belong to the cluster of A. Then  visit the samples within A's ε- neighbor successively. 

  ![Image 5](https://www.codeproject.com/KB/AI/5129186/b7571e61-dff9-4cd5-be40-f7017bba5825.Png)

- Visit the sample B within A's ε- neighbor. B is a core sample, visit the samples within B's ε- neighbor. 

  ![Image 6](https://www.codeproject.com/KB/AI/5129186/9fae958a-8548-47c5-a9e6-a121df7280b7.Png)

- Visit the sample C within B's ε- neighbor. C belongs to A and C is not a core point. Visit the other samples within B's ε- neighbor. 

  ![Image 7](https://www.codeproject.com/KB/AI/5129186/79c15ee4-8c9a-4063-816d-61ace2f71ac8.Png)

- Visit another sample D within B's ε- neighbor. D is a core point. Visit the samples within D's ε- neighbor. 

  ![Image 8](https://www.codeproject.com/KB/AI/5129186/26af5b2e-4f87-4107-a1a9-b0bdbc3e7f28.Png)

- Visit the sample E. E is not a core point. Now, there is not any point density-reachable to A. Stop clustering for A. 

  ![Image 9](https://www.codeproject.com/KB/AI/5129186/81917395-b54b-45ed-8515-82d0533a3f75.Png)

Hide   Copy Code

```
k = 0
unvisited = list(range(sample_num))               # samples which are not visited
while len(centers) > 0:
    visited = []
    visited.extend(unvisited)
    cores = list(centers.keys())
    # choose a random cluster center
    randNum = np.random.randint(0, len(cores))
    core = cores[randNum]
    core_neighbor = []                              # samples in core's neighbor
    core_neighbor.append(core)
    unvisited.remove(core)
    # merege the samples density-connectivity
    while len(core_neighbor) > 0:
        Q = core_neighbor[0]
        del core_neighbor[0]
        if Q in initial_centers.keys():
            diff = [sample for sample in initial_centers[Q] if sample in unvisited]
            core_neighbor.extend(diff)
            unvisited = [sample for sample in unvisited if sample not in diff]
    k += 1
    label[k] = [val for val in visited if val not in unvisited]
    for index in label[k]:
        if index in centers.keys():
            del centers[index]
```

### Parameter Estimation

DBSCAN algorithm requires 2 parameters **ε**, which specify how close points should be to each other to be considered a part of a cluster; and ***m\***, which specifies how many neighbors a point should have to be included  into a cluster. Take an example from wiki, we calculate distance from  each point to its nearest neighbor within the same partition as shown  below, we can easily determine **ε = 22**.

![Distribution of distances to the nearest neighbor of each point](https://www.codeproject.com/KB/AI/5129186/951e7da9-71b6-4d09-80c7-bcccc1729572.Png)

For parameter **m**, we calculate how many samples are within core point's **ε** neighborhood as shown below. We choose **m** = 129 because it is the first valley bottom.

![Distribution of distances to the number of neighbors](https://www.codeproject.com/KB/AI/5129186/88c5c6e8-3778-45f9-9bc7-5697c1b26b8e.Png)

## Conclusion and Analysis

DBSCAN has the ability to cluster nonspherical data but cannot  reflect high-dimension data. When the density is not well-distributed,  the clustering performance is not so good. The clustering performance  between KMeans and DBSCAN is shown below. It is easy to find that DBSCAN has better clustering performance than KMeans.

![Image 12](https://www.codeproject.com/KB/AI/5129186/c3bdb2b8-d500-4eb2-b65d-5a2fe18ae485.Png)

![Image 13](https://www.codeproject.com/KB/AI/5129186/00d6d108-389c-4c9a-abfe-933a61dbc010.Png)

![Image 14](https://www.codeproject.com/KB/AI/5129186/c2ea7cff-23f6-421b-9529-6539840a7210.Png)

![Image 15](https://www.codeproject.com/KB/AI/5129186/33c0e863-d566-4fe8-9e2f-5e9392c22470.Png)

The related code and dataset in this article can be found in [MachineLearning](https://github.com/DandelionLau/MachineLearning).