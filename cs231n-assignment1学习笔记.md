---
title: cs231n-assignment1学习笔记
date: 2023-10-11 20:40:34
categories:
- 课程笔记
tags:
- cs231n
---

# cs231n-assignment1学习笔记
此篇是cs231n课程课后作业1的笔记，包含对于几个分类器的简述，还有在分类过程包括数据处理、损失函数，梯度计算，模型训练和预测等中的常见操作，比如数据取样，批量训练，交叉验证，超参数选择，toy_model等，包括一些使用到的函数用法介绍，后一部分是重点。

## 1.分类器简介
### 1.1 knn
knn是指k近邻分类，原理就是找到空间距离图像最近的k张图像，然后在这k张图像中寻找出现次数最多的图像作为预测输出。
### 1.2 svm
![svm损失函数](./cs231n-assignment1学习笔记/svm.png)
### 1.3 softmax
![softmax损失函数](./cs231n-assignment1学习笔记/softmax.png)
### 1.4 two_layer_net
两层网络结构非常简单：输入->全连接层->relu函数->全连接层->softmax->results，这里我不再详细展开叙述。
## 2.常用方法及函数
### 2.1 concatenate，vstack和hstack
这三个函数都是进行数组连接的函数，详细如下：
+ concatenate函数起连接作用，它在已存在的维度上连接的数组
+ vstack函数是在垂直方向上进行堆叠，要求数组本身在列方向上的维度相同
+ hstack函数是在水平方向上进行堆叠，要求数组本身在行方向上的维度相同
在某些情况下它们功能相同，但也有差异，具体可以参见下面示例部分：
```python
import numpy as np
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)
# array([[1, 2],
#        [3, 4],
#        [5, 6]])

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.vstack([a, b])
# array([[1, 2],
#        [3, 4],
#        [5, 6]])

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]]).reshape(-1,1)
np.hstack([a, b])
# array([[1, 2, 5],
#        [3, 4, 6]])

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]]).reshape(-1,1)
np.concatenate([a, b])
# ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 2 and the array at index 1 has size 1
```
本次作业中第一次使用到此类函数是在CIFAR-10的数据加载中，此数据集有六个batch，前五个为数据集，最后一个为测试集，训练时需要将前五个batch连接成为一个训练集，使用到了concatenate函数。

还有一个使用到此类函数的情况是K折交叉验证的操作，我们将原始训练数据分为k个部分，然后选取其中一个部分为validation，其余四个部分作为训练集，在合并其余四个部分的过程中使用到了vstack函数，具体如下：
```python
X_train_cross = np.vstack(X_train_folds[0:i] +  X_train_folds[i+1:])
y_train_cross = np.vstack(y_train_folds[0:i] +  y_train_folds[i+1:])
```
### 2.2 argsort、argmax和bincount函数
这三个函数均是在knn的分类预测过程中使用到的，作用方法如下：
+ argsort函数返回数组从小到大的排序索引，可以使用axis=0或axis=1来指定按行或者按列进行排序
+ argmax函数返回数组最大值的索引，同样可使用axis指定按行或按列操作
+ bincount可以统计数组中非负整数的出现次数，注意不可包含负数和浮点数
+ 
本作业中使用如下：
```python
# 先找到距离最近的k个点的索引
closest_y = self.y_train[np.argsort(dists[i])[:k]].flatten()
# 然后统计k个索引中相同对象的个数并且找到出现次数最多的作为预测值
y_pred[i] = np.argmax(np.bincount(closest_y))
```
### 2.3 训练数据的获取与处理
这个部分主要记录分类过程中我们可能对数据进行的采集与处理：
+ 首先是对数据的采集，其实我们已经得到了X_train,y_train,X_test, y_test,但是我们可能还需要获取验证集和预测试集（小批量数据，方便对分类器进行测试），这时候我们可以建立mask方便对数据进行采样，如果需要随即获取部分数据，可以使用random.choice()函数，softmax分类中具体体现如下：
```python
 # subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]
```
+ reshape并且添加bias项，方便线性分类器(wx+b)的计算：
```python
# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

# add bias dimension and transform into columns
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
```
### 2.4 交叉验证过程
我在此将交叉验证视为一个拆分训练集进行训练和验证的过程，先说明为何要进行交叉验证？
+ 1.防止因为随意划分训练集和测试集导致的结果不一
+ 2.减小只使用部分数据进行训练和验证

它的作用主要是可以进行模型选择和模型评估。比如上s面提到的5折交叉验证，过程如下图：
![交叉验证](./cross-validation.png)
- 若是进行模型选择，也就是超参数选择，不同的参数就会有不同的预测准确率，如下图所示（相同列的五个点即是五折验证的不同结果）：
![交叉验证准确率](./cross-valiation-acc.png)
- 若是进行模型评估，只需按照求五个RMSE的平均值，以此作为模型评估的准则。

下面是knn中的交叉验证过程：
```python
for k in k_choices:
    accuracies = []
    for i in range(num_folds):
        X_train_cross = np.vstack(X_train_folds[0:i] +  X_train_folds[i+1:])
        y_train_cross = np.vstack(y_train_folds[0:i] +  y_train_folds[i+1:])
        X_test_cross = X_train_folds[i]
        y_test_cross = y_train_folds[i]
        print(X_train_cross.shape)
        classifier.train(X_train_cross, y_train_cross)
        dists = classifier.compute_distances_no_loops(X_test_cross)
        y_test_pred_cross = classifier.predict_labels(dists, k=k).reshape(-1, 1)

        num_correct_cross = np.sum(y_test_pred_cross == y_test_cross)
        accuracy = float(num_correct_cross) / len(y_test_cross)
        accuracies.append(accuracy)
    k_to_accuracies[k] = accuracies
```
### 2.5 超参数选择
这个过程其实比较比较简单，就是我们定义一个learning_rate和regularization_strength或者其他的超参数列表，我们循环遍历它们，计算每种情况下的validation上的准确率，进而找到较优的超参数。svm中的hyperparameters choosing过程如下：
```python
learning_rates = [1e-7, 3e-7, 5e-7, 7e-7]
regularization_strengths = [1e3, 3e3, 5e-3]
results = {}
best_val = -1  
best_svm = None

for lr in learning_rates:
    for reg in regularization_strengths:
        svm = LinearSVM()
        loss_hist = svm.train(X_train, y_train, lr, reg, num_iters=3000)
        y_pred_train = svm.predict(X_train)
        train_accuracy = np.mean(y_train == y_pred_train)
        y_val_pred = svm.predict(X_val)
        val_accuracy = np.mean(y_val == y_val_pred)
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_svm = svm
        results[(lr, reg)] = train_accuracy, val_accuracy
```
### 2.6 Toy model
toy_model其实类似于上面的预测试集获取，Toy model用于验证训练流程是否正常工作，是否能够收敛到一个合理的结果。它们可以帮助检查损失函数的设置是否正确，梯度计算是否准确，优化器的选择是否合适等。在two_payer_net中应用如下：
```python
input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

net = init_toy_model()
X, y = init_toy_data()
```
## 3.其他
其实作业一整个流程下来，还有一个艰难的任务是在梯度传播上，因为要计算权重的梯度，并且还要向量化计算，这是一个相对麻烦的任务，这部分内容我在上文中没有叙述，一是因为现在的神经网络其实不关注手动的梯度计算，二是这部分内容需要一定的数学基础，而我还有欠缺，如果后面有时间我会更新这部分的内容。

另一个我没有提到的但同样重要的内容是批量规范化，本文中对于输入的图片数据进行了减去均值的操作在某种意义上就是批量规范化的过程，当然真正的批量规范化将会在后续神经网络的内容中详细的解释。
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