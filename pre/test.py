import numpy as np
import scipy.io as sio


""" a = np.array(range(60))

b = a.reshape(3, 4, 5)
sio.savemat('./b.mat', {'b': b})
print("---------------------------")
c = b.transpose(2, 1, 0)
sio.savemat('./c.mat', {'c': c}) 
"""


for i in range(112):
    if i < 10:
        str_i = str(0) * 5 + str(i)
    elif i < 100:
        str_i = str(0) * 4 + str(i)
    else:
        str_i = str(0)*3+str(i)
    print(str_i)
