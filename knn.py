# -*- coding:utf-8 -*- 
# Author: Roc-J
'''
k近邻一个简单的示范，输入简单的数据点集，
在输入一个点，求这个点的三个最近邻点
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

X = np.array([
    [1, 1], [1, 3], [2, 2], [2.5, 5], [3, 1], [4, 2], [2, 3.5], [3, 3], [3.5, 4]
])

num_neighbors = 3

input_point = np.array([2.6, 1.7])

# plot the data
plt.figure()
plt.scatter(X[:, 0], X[:, 1], color = 'k', s=25, marker='o')
plt.title("dataset")

# create model
knn = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(X)

distance, indices = knn.kneighbors(input_point.reshape(1, -1))

# 打印输出k近邻点
print '\nk nearest neighbors'
for rank, index in enumerate(indices[0][:num_neighbors]):
    print str(rank+1) + '-->', X[index]

plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, color='k')
plt.scatter(X[indices][0][:][:, 0], X[indices][0][:][:, 1], marker='o', s=150, color='k', facecolors='none')
plt.scatter(input_point[0], input_point[1], color='k', s=50, marker='x')
plt.title('KNN ')
plt.show()

