# -*- coding:utf-8 -*- 
# Author: Roc-J

from sklearn.datasets import samples_generator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

# 生成样本数据
X, y = samples_generator.make_classification(n_informative=4, n_features=20, n_redundant=0, random_state=5)

# 特征选择器，选择k个最好的特征
selector_k_best = SelectKBest(f_regression, k=10)

# 随机森林分类器
classifier = RandomForestClassifier(n_estimators=50, max_depth=4)

# 构建机器学习流水线
'''
selector 特征选择器
classifier 随机森林分类器
'''
pipeline_classifier = Pipeline([('selector', selector_k_best), ('rf', classifier)])
# 可以进行参数的更新
pipeline_classifier.set_params(selector__k = 6, rf__n_estimators=25)

# train
pipeline_classifier.fit(X, y)

# output
prediction = pipeline_classifier.predict(X)
print "\nPredictions :\n", prediction

# measure
print "\nScores : ", pipeline_classifier.score(X, y)

# 打印被分类器
features_status = pipeline_classifier.named_steps['selector'].get_support()
selector_features = []
for count, item in enumerate(features_status):
    if item:
        selector_features.append(count)

print "\nSelected features (0-indexed): ", ', '.join([str(x) for x in selector_features])
