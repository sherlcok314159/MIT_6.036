**Perceptron**

```python
def perceptron(T,D):
    th = np.array([[0,0]]).T
    th0 = 0
    for t in range(1,T+1):
        for i in range(1,n+1):
            tem = np.sum(np.dot(th.T,x)) + th0
            if y * tem <= 0:
               th = th + y * x
               th0 = th0 + y 
    return (th,th0)
```

**Linear separability**

**Test error**

**Margin**

**Convergence Theorem**

**Prove it**

