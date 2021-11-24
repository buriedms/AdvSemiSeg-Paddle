import paddle
import numpy as np

a=np.array([1,2,3,4])
b=0.2
c=a<b
print(c.size)
print(c)
print(sum(c))