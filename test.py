import numpy as np


def get_training_dataset():
    raw_vecs = [[1,1], [0,0], [1,0], [0,1]]
    size_vecs = len(raw_vecs)
    input_vecs = np.reshape(raw_vecs, (size_vecs, 2))

    raw_labels = [1,0,0,0]
    size_labels = len(raw_labels)
    labels = np.reshape(raw_labels, (size_labels, 1))
    return input_vecs, labels


x, y = get_training_dataset()

z = np.reshape([0.1, 0.1], (2,1))

print(x[1, :])
print(z)
print(x[1, :].dot(z))



