# -*- coding:utf-8 -*- 
# Author: Roc-J

import json
import numpy as np
from Pearson_score import pearson_score

def find_similar_users(dataset, user, num_users):
    if user not in dataset:
        raise TypeError('User ' + user + 'not present in the dataset')

    # 计算所有用户的皮尔逊相关度
    scores = np.array([[x, pearson_score(dataset, user, x)] for x in dataset if user != x])
    # 按照得分降序排列
    scores_sorted = np.argsort(scores[:, 1])
    # 评分按照降序排列
    scored_sorted_dec = scores_sorted[::-1]

    # 提取k个最高分并返回
    top_k = scored_sorted_dec[:num_users]

    return scores[top_k]

if __name__ == '__main__':
    data_file = 'movie_ratings.json'
    with open(data_file, 'r') as f:
        data = json.loads(f.read())

    user = 'John Carson'
    print '\nUsers similar to ' + user +':\n'
    similar_users = find_similar_users(data, user, 3)
    print 'User  \t similar_score\n'
    for item in similar_users:
        print item[0], '\t', round(float(item[1]), 2)