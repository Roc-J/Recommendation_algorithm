# -*- coding:utf-8 -*- 
# Author: Roc-J

import numpy as np

def load_data(input_file):
    X = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data = [float(x) for x in line.split(',')]
            X.append(data)

    return np.array(X)

