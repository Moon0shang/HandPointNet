import numpy as np
import scipy.io as sio
from struct import *

""" a = np.array(range(60))

b = a.reshape(3, 4, 5)
sio.savemat('./b.mat', {'b': b})
print("---------------------------")
c = b.transpose(2, 1, 0)
sio.savemat('./c.mat', {'c': c}) 
"""


""" for i in range(112):
    if i < 10:
        str_i = str(0) * 5 + str(i)
    elif i < 100:
        str_i = str(0) * 4 + str(i)
    else:
        str_i = str(0)*3+str(i)
    print(str_i)
 """

""" with open('D:/Cache/Git/HandPointNet/data/cvpr15_MSRAHandGestureDB/P0/1/000000_depth.bin', 'rb') as f:
    # f.read(1)
    a = unpack('I', f.read(4))[0]
    b = unpack('I', f.read(4))[0]
    c = unpack('I', f.read(4))[0]
    d = unpack('I', f.read(4))[0]
    e = unpack('I', f.read(4))[0]
    g = unpack('I', f.read(4))[0]

    h = e - c
    i = g - d
    # j = []
    #for ii in range(h):
        #for jj in range(i):
           # j.append(struct.unpack('f', f.read(4)))

    buff = f.read(h*i*4)
    j = unpack('f'*h*i, buff)
    j1 = np.array(j, dtype=np.float32).reshape(h, i)
    print(j1)
    # j.reshape(h, i)
    print(j1.shape, j1.dtype)
    # for iii in j:
    # print(iii)
 """

a = np.zeros((4, 3))
print(a[3, 1])
