import paddle
from tqdm import tqdm
import numpy as np
from time import sleep

a = np.array([1, 2, 3, 4])
b = 0.2
c = a < b
print(c.size)
print(c)
print(sum(c))
a=[1, 2, 3, 4]
with tqdm(a) as t:
    for i in a:
        t.set_description(f'train:')
        t.set_postfix(data=f"{i:->8d}", loss='nohao', str='she呀', hellp='shide')
        sleep(1)
        t.update()
