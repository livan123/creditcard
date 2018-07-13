# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model.logistic import LogisticRegression
from pylab import *
from scipy.constants.constants import alpha

# predict_proba：主要是用来预测分类出现的概率；
# x_train = np.array([[1,2,3],[1,3,4],[2,1,2],[4,5,6],[3,5,3],[1,7,2]])
# y_train = np.array([3,3,3,2,2,2])
# 
# x_test = np.array([[2,2,2],[3,2,6],[1,7,4]])
# clf=LogisticRegression()
# clf.fit(x_train, y_train)
# 
# print(clf.predict(x_test))
# print(clf.predict_proba(x_test))
# 可以运算每个分类结果发生的概率：测试集中结果为（0,1）


import itertools
# 1）笛卡儿积：itertools.product
# for i in itertools.product('ABCD', repeat=2):
#     print(''.join(i), end=' ')
#     
# a = (1,2,3)
# b = ('A','B','C')
# c = itertools.product(a,b)
# for i in c:
#     print(i, end=' ')
    
# 2）排列：
# for i in itertools.permutations('ABCD', 2):
#     print(''.join(i), end=' ')


# 3）组合：
# for i in itertools.combinations('ABCD', 3):
#     print(''.join(i))

# 4）组合：
# for i in itertools.combinations_with_replacement('ABCD', 3):
#     print(''.join(i), end=' ')













