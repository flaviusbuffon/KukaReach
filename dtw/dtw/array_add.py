import numpy as np
# tp1 = (1, 2, 3, 4, 5, 6)
# tp2 = (2, 2, 3, 4, 5, 6)
# arr1 = np.array(tp1, dtype=float)
# arr1 = np.append(arr1,tp2)
# print(arr1)

x = np.empty(shape=[0, 3])
x = np.append(x, [[1,2,3]], axis = 0)
x = np.append(x, [[2,3,4]], axis = 0)
print(x)
