# -*- coding:utf-8 -*- 
# Author: Roc-J

import numpy as np
import json
from Pearson_score import pearson_score

def generate_recommendations(dataset, user):
    '''
    为给定用户生成电影推荐
    :param dataset:
    :param user:
    :return:
    '''
    if user not in dataset:
        raise TypeError('User' + user + ' not present in the dataset')

    total_scores = {}
    similarity_sums = {}

    # 计算该用户与其他用户的皮尔逊相关系数
    for u in [x for x in dataset if x != user]:
        similarity_score = pearson_score(dataset, user, u)

        if similarity_score <= 0:
            continue

        for item in [x for x in dataset[u] if x not in dataset[user] or dataset[user][x] == 0]:
            total_scores.update({item: dataset[u][item] * similarity_score})
            similarity_sums.update({item: similarity_score})

        if len(total_scores) == 0:
            return ['No recommendations possible']

        # 生成电影评分标准化
        movie_ranks = np.array([ [total/similarity_sums[item], item] for item, total in total_scores.items()])

        movie_ranks = movie_ranks[np.argsort(movie_ranks[:, 0])[::-1]]

        recommendations= [movie for _, movie in movie_ranks]
        return recommendations

if __name__ == '__main__':
    data_file = 'movie_ratings.json'
    with open(data_file, 'r') as f:
        data = json.loads(f.read())

    user = 'Michael Henry'
    print '\nRecommendations for ' + user + ':'
    movies = generate_recommendations(data, user)
    for i, movie in enumerate(movies):
        print i+1, '.', movie

    user = 'John Carson'
    print '\nRecommendations for ' + user + ':'
    movies = generate_recommendations(data, user)
    for i, movie in enumerate(movies):
        print i+1, '.', movie
