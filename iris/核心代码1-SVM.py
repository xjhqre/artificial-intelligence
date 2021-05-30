import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from sklearn import model_selection, svm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# 1.数据准备
# *************将字符串转为整型，便于数据加载***********************
def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]


# 1.1加载数据
data_path = 'iris.data'  # 数据文件的路径
data = np.loadtxt(
    data_path,  # 数据文件路径
    dtype=float,  # 数据类型
    delimiter=',',  # 数据分隔符
    converters={4: iris_type})  # 将第5列使用函数iris_type进行转换
# data为二维数组，data.shape=(150, 5)
print('数据集规模：\n', data.shape)
# 1.2数据分割
x, y = np.split(
    data,  # 要切分的数组
    (4, ),  # 沿轴切分的位置，第5列开始往后为y
    axis=1)  # 代表纵向分割，按列分割
x = x[:, 0:4]  # 在X中我们取前两列作为特征，为了后面的可视化。x[:,0:2]代表第一维(行)全取，第二维(列)取0~2
print("输入特征：\n", x)
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x,  # 所要划分的样本特征集
    y,  # 所要划分的样本结果
    random_state=1,  # 随机数种子
    test_size=0.3)  # 测试样本占比
# print('y_train:\n',y_train)
# print('y_test\n',y_test)
'''
#====绘制iris数据集==================
cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r']) 
plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark) # 样本点
#plt.scatter(x_test[:, 0], x_test[:, 1], c=np.squeeze(y_test), s=50, cmap=cm_dark) # 测试点
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()               #第0列的范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()               #第1列的范围
plt.xlabel('sepal length', fontsize=20)
plt.ylabel('sepal width', fontsize=20)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('iris data', fontsize=30)
plt.grid()
plt.show()
'''


# ======================
# **********************SVM分类器构建*************************
def classifier():
    clf = svm.SVC(C=10, kernel='rbf', gamma=0.1,
                  decision_function_shape='ovr')

    # clf = svm.SVC(
    #     C=5.0,  # 误差项惩罚系数,默认值是1
    #     kernel='linear',  # :线性核 kenrel="rbf":高斯核
    #     decision_function_shape='ovr')  # 决策函数,即一个类别与其他类别进行划分

    return clf


# 2.定义模型：SVM模型定义
clf = classifier()

print(y_train)


# ***********************训练模型*****************************
def train(clf, x_train, y_train):
    clf.fit(
        x_train,  # 训练集特征向量
        y_train.ravel())  # 训练集目标值


# 3.训练模型
train(clf, x_train, y_train)


# **************并判断a b是否相等，计算准确率acc的均值*************
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print('%s Accuracy:%.3f' % (tip, np.mean(acc)))


def print_accuracy(clf, x_train, y_train, x_test, y_test):
    # 分别打印训练集和测试集的准确率  score(x_train,y_train):表示输出x_train,y_train在模型上的准确率
    #print('trianing prediction:%.3f' %(clf.score(x_train, y_train)))
    #print('test data prediction:%.3f' %(clf.score(x_test, y_test)))

    # 原始结果与预测结果进行对比   predict()表示对x_train样本进行预测，返回样本类别
    show_accuracy(clf.predict(x_train), y_train, 'traing data')
    show_accuracy(clf.predict(x_test), y_test, 'testing data')

    # 计算决策函数的值，表示x到各分割平面的距离
    print('decision_function:\n', clf.decision_function(x_train))


# 4.模型评估
print_accuracy(clf, x_train, y_train, x_test, y_test)


# **************绘制输出结果**************
def draw(clf, x):
    iris_feature = 'sepal length', 'sepal width', 'petal lenght', 'petal width'
    # 开始画图
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # stack():沿着新的轴加入一系列数组
    #print('grid_test:\n', grid_test)

    # 输出样本到决策面的距离
    z = clf.decision_function(grid_test)
    #print('the distance to decision plane:\n', z)

    grid_hat = clf.predict(grid_test)  # 预测分类值 得到【0,0.。。。2,2,2】
    print('grid_hat:\n', grid_hat)
    grid_hat = grid_hat.reshape(x1.shape)  # reshape grid_hat和x1形状一致
    # 若3*3矩阵e，则e.shape()为3*3,表示3行3列

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r'])

    plt.pcolormesh(x1, x2, grid_hat,
                   cmap=cm_light)  # 区域图 # pcolormesh(x,y,z,cmap)这里参数代入
    # x1，x2，grid_hat，cmap=cm_light绘制的是背景。
    plt.scatter(x[:, 0],
                x[:, 1],
                c=np.squeeze(y),
                edgecolor='k',
                s=50,
                cmap=cm_dark)  # 样本点
    plt.scatter(x_test[:, 0],
                x_test[:, 1],
                c=np.squeeze(y_test),
                s=50,
                cmap=cm_dark)  # 测试点

    plt.xlabel(iris_feature[0], fontsize=20)
    plt.ylabel(iris_feature[1], fontsize=20)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('svm in iris data classification', fontsize=30)
    plt.grid()
    plt.show()


# 5.模型使用
draw(clf, x)
