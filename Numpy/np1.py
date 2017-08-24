import numpy as np

a = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32).reshape((3, 2))
b = np.arange(4, 10, dtype=np.int32).reshape((2, 3))

c_dot1 = np.dot(a, b)
c_dot2 = np.matmul(a, b)

print(a)
print(a)
print(b)
print(c_dot1)
print(c_dot2)

A = np.random.randint(0, 10, (5, 2), dtype=np.int32)

print(A)
print(A.mean(axis=1))
