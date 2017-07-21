import numpy as np

class a(object):
    def __init__(self, a):
        self.a = a

A = a(1)

b = []

b.append(A)

print(b[0].a)