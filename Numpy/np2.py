import numpy as np

A = np.arange(2, 14).reshape((3, 4))

print(A)
print(np.argmax(A, axis=1))
print(np.median(A, axis=0))
print(np.cumsum(A, axis=0))
print(np.diff(A, axis=0))

a = np.array([[1, 0, 0], [0, 2, 0], [1, 1, 0]])

print(a)
print(a.T.dot(a))

b = np.arange(10)

print(b.clip(0, 1))

print(b)

c = np.arange(0, 15).reshape((3, 5))
print(c.flat)
