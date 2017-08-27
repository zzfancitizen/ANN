import numpy as np

array = [[1, 2, 3], [2, 3, 4]]
npArray = np.array(array, dtype=np.int)
a = np.ones((3, 4), dtype=np.int16)
b = np.arange(10, 20, 2, dtype=np.int16).reshape((5, 1))
c = np.linspace(1, 9, 5, dtype=np.int16)

print(npArray)
print('dim: %i' % npArray.ndim)
print('shape: %s' % str(npArray.shape))
print('size: %i' % npArray.size)
print('type: %s' % npArray.dtype)

a = np.array([10, 20, 30, 40])
b = np.arange(4)

c = a + b

print(a ** 2)
