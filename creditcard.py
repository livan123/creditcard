# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
# from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis
from sklearn.linear_model.tests.test_passive_aggressive import random_state
# from ctypes.test import test_sizes
from random import shuffle
# 用户行为：哪些用户行为（）
# 判断用户的信用卡账单涉嫌欺诈
# 读取数据内容
data = pd.read_csv('creditcard.csv')
# 最后一列class为1则表示有欺诈行为，0为没有欺诈行为,所以此次的目的是对数据进行0、1的二分类问题，
# value_counts：计算相同数据出现的频率,此处为计算0和1各有多少个：
count_class = pd.value_counts(data['Class'], sort=True).sort_index()
# 此时会存在样本个数严重失衡的问题,需要过采样或下采样：
# 0:284315
# 1:492
# 特征工程：
# 1）去掉特征中多余的项Time；
# 2）amount的值需要归一化，将特别大的值划归到0-1之间；
from sklearn.preprocessing.data import StandardScaler
# -1表示系统自动计算得到的行，1表示1列；
# StandardScaler：数据标准化
# reshape：即形状重置，reshape(-1,1)表示所有的行数，全部整理成只有一列的样式；
data['normAmount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
# 删除两行，axis=1表示按照列删除，axis=0表示按照行删除；
data = data.drop(['Time','Amount'], axis=1)
# 3）对样本进行平衡性处理：
# 过采样：对少的样本再生成些；
# 下采样：随机选取类别为0的样本，使两者之间的样本一样；
# 下面是下采样的代码：
# ix是结合了iloc与loc两个函数，ix[1,1]:按照行列数进行取值，也可以ix["a","b"]:表示a行b列；
# xy为所有数据集：
x=data.ix[:,data.columns!='Class']
y=data.ix[:,data.columns=='Class']


# 统计异常值得个数
number_records_fraud = len(data[data.Class==1])
# 统计欺诈样本的下标，并变成矩阵的格式：
fraud_indices = np.array(data[data.Class==1].index)
# 记录正常值的下标：
normal_indices = data[data.Class==0].index
# 从正常值的索引中，选择和异常值相等个数的样本，保证样本的均衡：
# np.random.choice(a,size, replace, p):在a中以概率p随机选择size个数据，replace是指是否有放回；
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
# 将数据转换成数组：
random_normal_indices = np.array(random_normal_indices)
# fraud_indices：欺诈样本的下标；random_normal_indices：正常值数组；
# concatenate：数据库的拼接；axis=1：按照对应行的数据进行拼接；
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
# loc["a","b"]:表示第a行，第b列；
# iloc[1,1]:按照行列来索引，左式为第二行第二列；
# 获取下标所在行的所有列，即得到训练所需要的数据集：
under_sample_data = data.iloc[under_sample_indices,:]
# 将数据集按照class列进行分类，
x_undersample = under_sample_data.iloc[:, under_sample_data.columns!='Class']
y_undersample = under_sample_data.iloc[:, under_sample_data.columns=='Class']
# 计算正负比例为0.5

# 交叉验证
from sklearn.cross_validation import train_test_split  # 导入交叉验证模块的数据切分；
# 随机划分训练集和测试集：x为除了class之外的其他的值，y为最终的结果列；
# test_size:样本占比；
# 从原始集中获取到训练集与测试集：
# train_test_split：x,y按照test_size的尺寸随机提取数据，然后划分到四个数据集中；
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
# 数据平衡之后的数据中获取到训练集与测试集：
x_undersample_train, x_undersample_test, y_undersample_train, y_undersample_test = train_test_split(x_undersample, y_undersample, test_size=0.3, random_state = 0)
# 上面数据准备好了，需要进行模型选择和优化：

# 过采样的处理：
# features=data.ix[:,data.columns!='Class']
# label=data.ix[:,data.columns=='Class']
# features_train, features_test, label_train, label_test = train_test_split(features, label)
# overSample = SMOTE(random_state=0)
# new_features, new_label = overSample.fit_sample(features_train, label_train)
# new_features = pd.DataFrame(new_features)
# new_label = pd.DataFrame(new_label, columns=['Class'])

# 模型评估：对模型选择较合适的参数
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report
# 将清理后的数据传入到方法中：
def printing_Kfold_scores(x_train_data, y_train_data):
#   对数据进行5折分组；
    fold = KFold(len(y_train_data), 5, shuffle=False)
#   print(len(y_train_data))
    c_param_range = [0.01, 0.1, 1, 10, 100]# 惩罚力度参数；
    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns = ['C_parameter','Mean recall score'])
    results_table['C_parameter'] = c_param_range
#   形成一个两列的数据，c_parameter为第一列，
    j = 0
#     循环的使用五个惩罚力度：通过k折检验来确定逻辑回归函数在加入惩罚项时，他对应的参数为best_C；
    for c_param in c_param_range:
        print('-----------------------------')
        print('C_parameter:', c_param)
        print('-----------------------------')
        print('')
        
        recall_accs = []
        # 对fold中进行遍历，fold中共有五组数据，start=1：下标从1开始
        # enumerate的作用是将fold数据结构组合为一个序列索引，同时列出数据以及下标；
        for iteration, indices in enumerate(fold, start=1):
            # print(iteration) 表示第几次循环；
            # print(indices) indices中返回值有两个，即两组值得下标，第一个为抽样后的剩余数据，用来作为训练集，一般占5分之4；第二个为抽样的数据，用来作为验证集，一般占5分之1；
            # 构建逻辑回归的样式，带有l1惩罚项；
            lr = LogisticRegression(C = c_param, penalty='l1')
            # 将数据放入模型中进行调整
            # x_train_data.iloc[indices[0],:]：4/5数据所对应的训练数据；1/5数据所对应的测试数据；
            # 将多维数据降为一维：
            #   ravel()：返回的是视图，修改对原数据有影响；
            #   flatten()：返回的是复制的内容，修改对原数据没有影响；
            lr.fit(x_train_data.iloc[indices[0],:], y_train_data.iloc[indices[0],:].values.ravel())
            # 利用交叉验证进行预测:利用取出的数据进行验证，indices中的第二维是抽取出来的1/5的数据，用来进行交叉验证的；
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)
            # print(y_pred_undersample) 验证出来的
            # 验证召回率：正确的结果有多少被给出了；
            # a=y_train_data.iloc[indices[1],:].values：总的正确结果数：
            # b=y_pred_undersample：预测结果是正确的：sum(a,b一致)
            # recall_acc = sum(a,b一致)/sum(a)；
            # 准确率：给出的结果有多少是正确的；
            recall_acc = recall_score(y_train_data.iloc[indices[1],:].values, y_pred_undersample)
            recall_accs.append(recall_acc)
#             print('Iteration', iteration, ':recall score=', recall_acc)
            # 在某一惩罚力度下，5组数据形成的集合，最终求平均值；
        results_table.ix[j, 'Mean recall score'] = np.mean(recall_accs)
        j+=1
#         print('')
#         print('Mean recall score', np.mean(recall_accs))
#         print('')
        # 最大值所对应的索引值   
    # print(results_table) 得到的是一个表格，记录了惩罚系数与召回率均值；
    # idxmax：获取召回率最大值；
    best_c = results_table.ix[results_table['Mean recall score'].astype('float64').idxmax()]['C_parameter']
    print('*******************************************************************')
    print('Best model to choose from cross validation is with C parameter =', best_c)
    print('*******************************************************************')
    return best_c
# 到目前为止，确定好模型需要的参数        
best_c = printing_Kfold_scores(x_undersample_train, y_undersample_train) 

# 参数确定好了，模型也就确定了，接下来的工作就是优化模型的效果；

import itertools 
from pylab import *
# 混淆矩阵：
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, s=1):#, cmap=plt.cm.Blues
    # 绘制热图
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # , cmap=cmap
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max()/2 
    
    # itertools.product：笛卡儿积；
    # 对括号中的两个值进行笛卡尔求积；
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # 文字显示内容：
        plt.text(j, i, cm[i, j], horizontalalignment="center")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
   
# 构建的模型：    
lr = LogisticRegression(C = best_c, penalty='l1')
lr.fit(x_undersample_train, y_undersample_train.values.ravel())
# 得到预测结果
# 使用默认阈值时：
# y_pred_undersample = lr.predict(x_undersample_test.values)
# 需要调节阈值时：predict_proba:返回的是n行k列的数组，第i行j列的数值是模型预测第i个样本为某个标签的概率，并且每一行的概率之和为1；
y_pred_undersample = lr.predict_proba(x_undersample_test.values)
# 使用混淆矩阵来判断数据预测的准确性；
print(y_pred_undersample[:,1])

# 这个阈值是指逻辑回归函数的分类阈值，使用predict_proba计算各个分类结果发生的概率，
thresh = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
plt.figure(figsize=(10,10))
j=1
for i in thresh:
    y_test_prediction = y_pred_undersample[:,1]>i
    # 返回的y_test_prediction为true：1或者false：0；
    print(y_test_prediction)
    plt.subplot(3,3,j)
    j+=1
    cnf_matrix = confusion_matrix(y_undersample_test, y_test_prediction)
    np.set_printoptions(precision=2)
    print("Recall metric in the testing dataset:", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# 首先：考虑是不是数据有问题：
# 1）直接使用原始数据获取新的最优惩罚系数，得到混淆矩阵，效果也不明显；
# 2）样本的欠采样情况下，查看模型的分类效果；
# 3）样本的过采样情况下，查看模型的分类效果；
# 其次：考虑是不是模型存在问题：
# 1）考虑惩罚系数：按照下采样获取到惩罚系数，得到混淆矩阵，会产生一个预测效果，结果并不理想，所以需要再考虑优化；
# 2）考虑阈值：阈值即为不同的分类阈值，即逻辑回归在多大概率上认为事件是0，多大概率上认为事件是1，通常默认为0.5；

    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix, classes =class_names, title = 'Confusion matrix>=%s'%i)
plt.show()


