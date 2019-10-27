# HW2

##1 Feedforward: Building a ReLU Neural Network

###1.1

![ezgif.com-crop.png](https://i.loli.net/2019/10/18/kxDdVMG5wyhg3mi.png)

### 1.2

$$ Output = ReLU(v_1*h_1+v_2*h_2+v_3*h_3+b_i) $$

$$ h_i = ReLU(w_{1i}*x_1+w_{2i}*x_2+b) $$

Note:

$x_i$ stands for the $i^{th}$ input

$h_i$ stands for the output for the $i^{th}$ unit of hidden layer

$w_{ij}$ stands for the weight from $x_i$ to $h_j$

$v_i$ stands for the weight from $h_i$ to the output unit

$b_i$ stands for the bias for the $i^{th}$ unit in hidden layer

$b$ stands for the bias for the output

 

###1.3

```python
import numpy as np
def ReLU(x):
    return maximum(0,x)

def ff_nn__ReLu(x,w,v,b_1,b):
    
    a = np.dot(x,w) + b_1
    h = ReLU(a)
    y = ReLU(np.dot(h,v) + b)

    return np.array(y)
```



### 1.4

result:

```python
array([[2. ],
       [1. ],
       [0.5]])
```



## 2 Gradient Descent

### 2.1

 $\frac{\partial f}{\partial x} = -3x^3+100*2(y^2-x)*(-1) = -3x^2-200(y^2-x)$

 $\frac{\partial f}{\partial y} = 100*2(y^2-x)*2*y=400y(y^2-x)$

### 2.2

![WechatIMG2.png](https://i.loli.net/2019/10/18/N3s7Lp1ha54dt2c.png)

### 2.3



### 2.4

