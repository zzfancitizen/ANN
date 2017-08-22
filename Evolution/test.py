import numpy as np

# a = np.random.randint(0, 2, (1, 10)).repeat(10, axis=0)
# b = np.random.randint(0, 2, size=10).astype(np.bool)
#
# i_ = np.random.randint(0, 10, size=1)

print(np.random.randint(*[32, 126], size=(300, 12)).astype(np.int8))