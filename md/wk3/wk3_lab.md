*Q1:What's tsv*
***
***Feature Engineering For Food Reviews***

**1.Read and Transform data**

```python
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer

train_df = pd.read_csv("...tsv",sep = 
"\t",nrows = 100)
vectorizer = CountVectorizer()
train_data = np.transpose(vectorizer.fit_transform(train_df["summary"]).toarray())
train_labels = np.array([list(train_df["sentiment"])])
```

由于是不定长的文本，所以需要采用*Bag of words*，将文本转换为向量的方式


**2.Perceptron Algorithm**
```python
# 普通分类算法
def perceptron(train_data, train_labels, params={}):
    T = params.get("T", 100)
    d, n = np.shape(train_data)
    # initialize th,th0
    th = np.array([[0.0] * d]).T
    th0 = 0.0
    for t in range(1, T + 1):
        for i in range(1, n + 1):
            train_x = train_data[:, i - 1 : i]
            train_y = train_labels[:, i - 1 : i]
            tem = np.sum(np.dot(th.T, train_x)) + th0
            if train_y * tem <= 0:
                th += train_y * train_x
                th0 += train_y
    return (th.T, th0)
```

**3.Averaged_Perceptron**
```python
def averaged_perceptron(train_data, train_labels, params={}):
    T = params.get("T", 100)
    d, n = np.shape(train_data)
    # initialize th,th0
    th = np.array([[0.0] * d]).T
    th0 = 0.0
    ths = np.array([[0.0] * d]).T
    th0s = 0.0
    for t in range(1, T + 1):
        for i in range(1, n + 1):
            train_x = train_data[:, i - 1 : i]
            train_y = train_labels[:, i - 1 : i]
            tem = np.sum(np.dot(th.T, train_x)) + th0
            if train_y * tem <= 0:
                th += train_y * train_x
                th0 += train_y
            ths += th
            th0s += th0
    return (ths.T / (n * T), th0s / (n * T))
```
**4.Test Data and Labels**
```python
# test_data + test_labels
test_df = pd.read_csv("E:\\lab3_data\\lab3_data\\reviews.tsv", sep="\t", nrows=322)
test_df = test_df.loc[151:]
test_data = test_df["summary"]
vectorizer = CountVectorizer()
test_data = np.transpose(vectorizer.fit_transform(test_data).toarray())
test_labels = np.array([list(train_df["sentiment"])])
```
因为测试数据最好不要与训练数据重合，所以进行了切片处理

**5.Evaluate Classifier**
```python
def eval_classifier(th_out, th0_out, test_data, test_labels):
    score_sum = 0
    d, n = np.shape(test_data)
    res = np.dot(th_out, test_data) + th0_out
    score_sum += np.sum(test_labels == np.sign(res))
    return score_sum / n
```

**6.Evaluate Learning Algorithm**
```python
def eval_algorithm(learner, test_data, test_labels):
    th_out, th0_out = learner(train_data, train_labels)
    return eval_classifier(th_out, th0_out, test_data, test_labels)
```

**7.Evaluate Learning Algorithm With A Fixed DataSet**
```python
# leave one out
def xval_algorithm(learner, train_data, train_labels, k):
    # shuffle train_data
    d1, n1 = np.shape(train_data)
    d2, n2 = np.shape(train_labels)
    train_data_div = np.array_split(train_data, n1, axis=1)
    train_labels_div = np.array_split(train_labels, n2, axis=1)
    score_sum = 0.0
    for i in range(k):
        train_data = np.concatenate(
            train_data_div[:i] + train_data_div[i + 1 :], axis=1
        )
        train_labels = np.concatenate(
            train_labels_div[:i] + train_labels_div[i + 1 :], axis=1
        )
        test_data = train_data_div[i]
        test_labels = train_labels_div[i]
        th_out, th0_out = learner(train_data, train_labels)
        score_sum += eval_classifier(th_out, th0_out, test_data, test_labels)
    return score_sum / k
```
这是留一法，先将数据集洗牌，然后平均分割，将一个数据用来验证，其他的用来训练

但是缺点是耗的时间过多，这里采取*10-fold cross validation*

**8.10-Fold Cross Validation**
```python
# 10-fold-cross validation
def xeval_algorithm_10(learner, train_data, train_labels):
    train_data_div = np.array_split(train_data, 10, axis=1)
    train_labels_div = np.array_split(train_labels, 10, axis=1)
    score_sum = 0
    for i in range(10):
        train_data = np.concatenate(
            train_data_div[:i] + train_data_div[i + 1 :], axis=1
        )
        train_labels = np.concatenate(
            train_labels_div[:i] + train_labels_div[i + 1 :], axis=1
        )
        test_data = train_data_div[i]
        test_labels = train_labels_div[i]
        th_out, th0_out = learner(train_data, train_labels)
        score_sum += eval_classifier(th_out, th0_out, test_data, test_labels)
    return score_sum / 10
```

这里与留一法的唯一区别就是分为10组即可，其他不变

**9.Testing**
```python
if __name__ == "__main__":
    print(eval_algorithm(perceptron, test_data, test_labels))
    # 0.4152046783625731
    print(eval_algorithm(averaged_perceptron, test_data, test_labels))
    # 0.5321637426900585

    print(eval_algorithm(perceptron, train_data, train_labels))
    # 1.0
    print(eval_algorithm(averaged_perceptron, train_data, train_labels))
    # 1.0

    print(xval_algorithm(perceptron, train_data, train_labels, 120))
    # 0.6416666666666667
    print(xval_algorithm(averaged_perceptron, train_data, train_labels, 120))
    # 0.7

    print(xeval_algorithm_10(perceptron, train_data, train_labels))
    # 0.5933333333333333
    print(xeval_algorithm_10(averaged_perceptron, train_data, train_labels))
    # 0.7066666666666667
```