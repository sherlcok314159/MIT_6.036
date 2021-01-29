### Introduction to numpy 
```python
import numpy as np
```
***
*1.创建数组*
```python
#维数与中括号对数有关
print(np.array([1,2,3]))
#1 dimension [1,2,3]
print(np.array([[1,2,3]]))
#2 dimensions [[1,2,3]]
```
***

*2.转置矩阵*
```python
#不改变维度，一维不变
a = np.array([1,2,3])
print(np.transpose(a))
#[1,2,3]
b = np.array([[1,2,3]])
print(b.T)
#[[1]
# [2]
# [3]]
```

*3.矩阵属性*
```python
#1 dimension
a = np.array([1,2,3])
print(np.shape(a))
#(3,) 
#2 dimensions
b = np.array([[1,2,3],[4,5,6]])
print(np.shape(b))
#(2,3)
```

*4.矩阵相乘*
```python
a = np.array([[1,2,3]])
b = np.array([[4,5,6]])
print(np.dot(a.T,b))
#[[ 4  5  6]
# [ 8 10 12]
# [12 15 18]]
#注意满足列数等于行数
```

*5.返回判断正负性*
```python
#正数返回1，零返回0，负数返回-1
print(np.sign(1))
# 1
print(np.sign(0))
# 0
print(np.sign(-1))
# -1
```

*6.求和*
```python
a = np.array([[1,2,3],[4,5,6]])
print(np.sum(a))
#21 不加参数为全部求和
print(np.sum(a,axis = 0))
#[5,7,9]
#看最外面的中括号，里面的元素为[1,2,3],[4,5,6],两个列表之间进行操作
print(np.sum(a,axis = 1))
#[6,15] 
#看最里面的中括号，里面的元素是不是一个为1，2，3.另一个是4，5，6，那元素之间进行求和，两者互不干扰
#同样的axis原则适用于np其他函数
```

*7.索引与切片*
```python
a = np.array([[1,2,3],[4,5,6]])
print(a[0])
#[1,2,3] 
#可以理解为两个小列表作为一个大列表的元素
print(a[0:1])
#[[1,2,3]]
print(a[:,2:])
#[[3]
# [6]]
#注意，如果中间加逗号，则意味着前面对行进行操作，后面对列进行操作
print(a[:,0])
#[1,4]
#注意，如果单独用整数索引或者切片，数组会保持原来的维度，但如果两者混用，则会导致降维
```

*8.行向量与列向量*

```python
#row vector
a = np.array([[1,2,3,4]])
#column vector
b = np.array([[1],[2],[3]])
#接收一个列表变为行向量
t = np.array([value_list])
#接收一个列表变为列向量
q = t.T or np.transpose(t)
```

*9.数组的长度*
```python
a = np.array([[1,2,3]])
print(np.sum(a * a)**0.5)
#3.7416573867739413 
#与求向量的模类似
```

*10.*