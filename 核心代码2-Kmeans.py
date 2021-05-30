import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics
from sklearn.cluster import DBSCAN, KMeans

# 1.准备数据
iris = datasets.load_iris()
X = iris.data[:, :4]  # 表示取特征空间中的4个维度
print('鸢尾花数据集规模：', X.shape)
# print("目标标签:\n",iris.target)  #鸢尾花数据集的目标

# 绘制数据分布图（鸢尾花数据集）
plt.scatter(X[:, 0], X[:, 1], c="black", marker='o', label='see')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()

# 2.配置模型
estimator = KMeans(n_clusters=4)  # 构造聚类器

# 3.训练模型
estimator.fit(X)  # 聚类

# 4.评估模型
label_pred = estimator.labels_  # 获取聚类标签
print("聚类标签:\n", label_pred)
# 平均轮廓系数为所有样本的轮廓系数的平均值
# 单个样本的轮廓系数在[-1,1]之间，越大越好，值为负说明在该样本上聚类出错
print("平均轮廓系数：", metrics.silhouette_score(X, label_pred, metric='euclidean'))

# 5.输出绘制聚类结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]

plt.scatter(x0[:, 0], x0[:, 1], c="green", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="blue", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="red", marker='+', label='label2')

plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()
