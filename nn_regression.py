# -*- coding:utf-8 -*- 
# Author: Roc-J

import numpy as np

import matplotlib.pyplot as plt
from sklearn import neighbors

# 生成正态分布的数据
amplitude = 10
num_points = 100
X = amplitude * np.random.rand(num_points, 1) - 0.5*amplitude

# 加入噪声
y = np.sinc(X).ravel()
y += 0.2 * (0.5 - np.random.rand(y.size))

# plot data
plt.figure()
plt.scatter(X, y, s=40, c='k', facecolors='none')
plt.title('Input data')

plt.show()

# 输入数据翻倍10
x_values = np.linspace(-0.5 * amplitude, 0.5* amplitude, 10* num_points) [:, np.newaxis]

n_neighbors = 8

knn_regressor = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
y_values = knn_regressor.fit(X, y).predict(x_values)

# 用原来的数据来训练， 预测10倍后的数据
plt.figure()
plt.scatter(X, y, s=40, c='k', facecolors='none', label='input data')
plt.plot(x_values, y_values, c='k', linestyle='--', label='predicted values')
plt.xlim(X.min() - 1.0, X.max() + 1.0)
plt.ylim(y.min() - 0.1, y.max() + 0.1)
plt.legend()
plt.title('k Nearest Neighbors Regressor')
plt.show()