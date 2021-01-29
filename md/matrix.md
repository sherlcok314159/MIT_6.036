### Multiplication
**1.Hadamard product（哈达玛积)**

**形式：**

***A * B***

**定义：**

>*a,b 都是 i x j 的同阶矩阵,设 c 是两者哈达玛积后的结果*
>
>*c(i) = a(i) * b(i)*

*e.g.* 

*a = np.array([1,2,3]),b = np.array([4,5,6])*

*c = a * b = [4,10,18]*(当然，输出的时候肯定看不到逗号)

*a' = np.array([[1,2,3],[4,5,6]]),b' =np.array([[1,-1,1],[6,7,8]])*

*c' = a' * b' = [[1,-2,3],[24,35,48]]*

***
**2. Dot product（点积)**

**形式：**

*c = a·b = （a^T）\* b* 

*a·b = ||a|| * ||b|| * cos(a,b)* (a,b之间的夹角)

前者为两个向量，后者 *a^T* 指示矩阵a的转置，转置的原因是因为需要满足矩阵相乘的原则：

**前一个矩阵的列数等于后一个矩阵的行数**

**注意，这里的点积结果为实数，而非矩阵，即两者的shape为(1,n),(n,1),结果为1 x 1的矩阵，所以要变成数字，还需要np.sum**

e.g.

```python
import numpy as np

th = np.array([[0], [0]])
data = np.array([[-3, 2, -1, 1], [3, 2, 1, 5]])
print(np.sum(np.dot(th.T, data[:, 0:1])))
#0
```

### Basic Rules
