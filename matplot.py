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

# predict_proba：预测的应用：
# x_train = np.array([[1,2,3],[1,3,4],[2,1,2],[4,5,6],[3,5,3],[1,7,2]])
# y_train = np.array([3,3,3,2,2,2])
# 
# x_test = np.array([[2,2,2],[3,2,6],[1,7,4]])
# clf=LogisticRegression()
# clf.fit(x_train, y_train)
# 
# print(clf.predict(x_test))
# print(clf.predict_proba(x_test))
# 可以运算每个分类结果发生的概率：测试集中结果为（0,1）：

# 创建一个图：
# 折线图：
# squares = [1,4,9,16,25]
# input_values = [1,2,3,4,5]
# plt.plot(input_values, squares, linewidth=5)
# plt.title('Square Number', fontsize=24)
# plt.xlabel('value', fontsize=14)
# plt.ylabel('Square of Value', fontsize=14)
# plt.tick_params(axis='both', labelsize=14)
# plt.show()
# 散点图：
# x_values = [1,2,3,4,5]
# y_values = [1,4,9,16,25]
# plt.scatter(x_values,y_values,s=100)
# plt.title('Square Numbers', fontsize=24)
# plt.xlabel('Value', fontsize=14)
# plt.ylabel('Square of Vaule', fontsize=14)
# plt.tick_params(axis='both', labelsize=14)
# plt.show()
# 自动计算数据：
# x_values = list(range(1, 1001))
# y_values = [x**2 for x in x_values]
# plt.scatter(x_values, y_values, s=1, edgecolors='none', cmap=plt.cm.Blues)
# plt.title('Square Number', fontsize=24)
# plt.xlabel('Value', fontsize=14)
# plt.ylabel('Square of Value', fontsize=14)
# plt.axis([0, 1100, 0, 1100000])
# plt.tick_params(axis='both', labelsize=14)
# plt.show()
# 随机漫步：
# class RandWalk(object):
#     def __init__(self, count=5000):
#         self.count = count
#         self.x_list = [0]
#         self.y_list = [0]
#         
#     def fill_walk(self):
#         while len(self.x_list)<self.count:
#             x_step = self.fill_step()
#             y_step = self.fill_step()
#             if x_step==0 or y_step==0:
#                 continue
#             next_x = self.x_list[-1] + x_step
#             next_y = self.y_list[-1] + y_step
#             
#             self.x_list.append(next_x)
#             self.y_list.append(next_y)
#             
#     def fill_step(self):
#         direction = choice([-1, 1])
#         distance = choice([2,4,6,8,10])
#         return direction*distance
#     
# rw = RandWalk()
# rw.fill_walk()
# point_numbers = list(range(rw.count))
# plt.figure(dpi=128, figsize=(10,6))
# plt.scatter(rw.x_list, rw.y_list, c=point_numbers, cmap=plt.cm.Blues, s=1)
# plt.scatter(0, 0, c='green', edgecolors='none', s=10)
# plt.scatter(rw.x_list[-1], rw.y_list[-1], c='red', edgecolors='none', s=10)
# 
# plt.axes().get_xaxis().set_visible(False)
# plt.axes().get_yaxis().set_visible(False)
# 
# plt.show()

# 其他图形为：
# bar(x,y,marker='s',color='r'):柱状图；
# hist(data, 40, normed=1, histtype='bar', facecolor='yellowgreen', alpha=0.75):直方图
# 设置x,y的坐标值：xlim(-2.5, 2.5);ylim(-1, 1)
# 显示中文和负号代码如下：
# plt.rcParams['font.sas-serig']=['SimHei']:显示中文标签；
# plt.rcParams['axes.unicode_minus']=False:用来正常显示负号；

# 创建子图：
# x=[1,2,3,4]
# y=[3,5,10,25]
# fig = plt.figure()
# ax1 = fig.add_subplot(231)
# plt.plot(x,y,marker='D')
# plt.sca(ax1)
# 
# ax2 = fig.add_subplot(232)
# plt.scatter(x,y,marker='s',color='r')
# plt.sca(ax2)
# plt.grid(True)
# 
# ax3 = fig.add_subplot(233)
# plt.bar(x,y,0.5,color='c')
# plt.sca(ax3)
# 
# ax4 = fig.add_subplot(234)
# mean=0
# sigma=1
# data=mean+sigma*np.random.randn(10000)
# plt.hist(data,40,normed=1,histtype='bar',facecolor='yellowgreen', alpha=0.75)
# plt.sca(ax4)
# 
# m = np.arange(-5.0, 5.0, 0.02)
# n = np.sin(m)
# ax5 = fig.add_subplot(235)
# plt.plot(m,n)
# plt.sca(ax5)
# 
# ax6 = fig.add_subplot(236)
# xlim(-2.5, 2.5)
# ylim(-1,1)
# plt.plot(m,n)
# plt.sca(ax6)
# plt.grid(True)
# 
# plt.show()

# 热图：
# x=[[1,2],[3,4],[5,6]]
# fig = plt.figure()
# ax = fig.add_subplot(231)
# ax.imshow(x)
# 
# ax = fig.add_subplot(232)
# im = ax.imshow(x, cmap=plt.cm.gray)
# 
# ax = fig.add_subplot(233)
# im = ax.imshow(x, cmap=plt.cm.spring)
# plt.colorbar(im)
# 
# ax = fig.add_subplot(234)
# im = ax.imshow(x, cmap=plt.cm.summer)
# plt.colorbar(im, cax=None, ax=None, shrink=0.5)
# 
# ax = fig.add_subplot(235)
# im = ax.imshow(x, cmap=plt.cm.autumn)
# plt.colorbar(im, shrink=0.5, ticks=[-1,0,1])
# 
# ax = fig.add_subplot(236)
# im = ax.imshow(x, cmap=plt.cm.winter)
# plt.colorbar(im, shrink=0.5)
# 
# plt.show()


# 矩阵颜色图：
# def draw_heatmap(data,xlabels,ylabels):
#     cmap=cm.get_cmap('rainbow',1000)
#     figure=plt.figure(facecolor='w')
#     ax=figure.add_subplot(1,1,1,position=[0.1,0.15,0.8,0.8])
#     ax.set_yticks(range(len(ylabels)))
#     ax.set_yticklabels(ylabels)
#     ax.set_xticks(range(len(xlabels)))
#     ax.set_xticklabels(xlabels)
#     vmax=data[0][0]
#     vmin=data[0][0]
#     for i in data:
#         for j in i:
#             if j>vmax:
#                 vmax=j
#             if j<vmin:
#                 vmin=j
#     map=ax.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto',vmin=vmin,vmax=vmax)
#     cb=plt.colorbar(mappable=map,cax=None,ax=None,shrink=0.5)
#     plt.show()
#             
# a=np.random.rand(10,10)
# print(a)
# xlabels=['A','B','C','D','E','F','G','H','I','J']
# ylabels=['a','b','c','d','e','f','g','h','i','j']
# draw_heatmap(a,xlabels,ylabels)
















