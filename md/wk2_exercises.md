### A real ML process

**1.Choose a Learning Algorithm and Implement**
```python
import numpy as np

def perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)#循环次数
    #shape of data
    d,n = np.shape(data)
    #initialize th,th0
    th = np.array([[0.] * d]).T#初始化th,注意与data保持形式相符
    th0 = 0.#浮点数
    for t in range(1,T+1):
        for i in range(1,n+1):
            train_x = data[:,i-1:i]
            train_y = labels[:,i-1:i]
            tem = np.sum(np.dot(th.T,train_x)) + th0
            if train_y * tem<= 0.:
                th += train_y * data[:,i-1:i]
                th0 += train_y#注意y不是数字，是数组
    return (th,th0)
```
**2.Optimize the learner**
```python
import numpy as np

def averaged_perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)

    #initialize th,th0
    d,n = np.shape(data)
    th = np.array([[0.0]*d]);th0 = 0.0
    ths = np.array([[0.0]*d]);th0s = 0.0
    for t in range(1,T+1):
        for i in range(1,n+1):
            y_i = labels[:,i-1:i]
            x_i = data[:,i-1:i]
            tem = np.dot(th,x_i) + th0
            if y_i * tem <= 0:
                th += (y_i * x_i).T
                th0 += y_i
            ths = ths + th
            th0s = th0s + th0
    return (ths/(n*T)).T,th0s/(n*T)
```
**3.Evaluate a classifier**
```python
import numpy as np

def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th_out,th0_out = learner(data_train,labels_train)
    d,n = np.shape(data_test)
    return score(data_test,labels_test,th_out,th0_out)/n
#returns the percentage correct on a new testing set as a float between 0. and 1..
```
**4.Evaluate a learning algorithm with a data source**
```python
import numpy as np

def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    score_sum = 0
    for i in range(it):
        data_train,labels_train = data_gen(n_train)
        data_test,labels_test = data_gen(n_test)
        score_sum += eval_classifier(learner, data_train, labels_train, data_test, labels_test)
    return score_sum/it
#data_gen - a data generator, call it with a desired data set size; returns a tuple (data, labels)
```
**5.Evaluating a learning algorithm with a fixed dataset**
```python
import numpy as np

def xval_learning_alg(learner, data, labels, k):
    #cross validation of learning algorithm
    #shuffle data and labels
    data_div,labels_div = np.array_split(data,k,axis = 1),np.array_split(labels,k,axis = 1)
    score_sum = 0
    for j in range(k):
        data_train = np.concatenate(data_div[:j]+data_div[j+1:],axis = 1)
        labels_train = np.concatenate(labels_div[:j]+labels_div[j+1:],axis = 1)
        data_test,labels_test = data_div[j],labels_div[j]
        score_sum += eval_classifier(learner, data_train, labels_train, data_test, labels_test)
    return score_sum/k
``` 
**6.Testing with pflip**

*Use your eval_learning_alg and the gen_flipped_lin_separable function in the code file to evaluate the accuracy of perceptron vs. averaged_perceptron. gen_flipped_lin_separable is a wrapper function that returns a generator - flip_generator, which can be called with an integer to return a data set and labels. Note that this generates linearly separable data and then "flips" the labels with some specified probability (the argument pflip); so most of the results will not be linearly separable. You can also specifiy pflip in the call to the generator wrapper function.*

```python
print(eval_learning_alg(perceptron, gen_flipped_lin_separable, 100, 100))
```
****
**Train on New data**

*pflip = 0.1*

*Accuracy for perceptron:0.7275000000000001*

*Accuracy for  averaged perceptron:0.8162499999999999*

*pflip = 0.25*

*Accuracy for perceptron:0.583125*

*Accuracy for averaged perceptron:0.6275000000000002*

**Conclusion:Averaged perceptron is better than perceptron**
***
**Train on train data**

*pflip = 0.1*

*Accuracy for perceptron = 0.818125*

*Accuracy for averaged perceptron = 0.8756250000000001*

*pflip = 0.25*

*Accuracy for perceptron = 0.6714999999999998*

*Accuracy for averaged perceptron = 0.7135000000000002*

**Conlusion:Learning algorithms have memory for training data**