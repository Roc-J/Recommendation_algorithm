# -*- coding:utf-8 -*- 
# Author: Roc-J

import json
import numpy as np

def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('User ' + user1 + ' not present in the dataset')
    if user2 not in dataset:
        raise TypeError('User ' + user2 + ' not present in the dataset')

    # 提取两个用户都评论过的电影
    rated_by_both = {}

    for item in dataset[user1]:
        rated_by_both[item] = 1

    if len(rated_by_both) == 0:
        return 0

    squared_diffenrences = []
    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diffenrences.append(np.square(dataset[user1][item] - dataset[user2][item]))

    return 1/(1+np.sqrt(np.sum(squared_diffenrences)))

if __name__ == '__main__':
    data_file = 'movie_ratings.json'

    with open(data_file, 'r') as f:
        data = json.loads(f.read())

    user1 = 'John Carson'
    user2 = 'Michelle Peterson'

    print '\nEuclidean score'
    print euclidean_score(data, user1, user2)