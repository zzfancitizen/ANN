import numpy as np


class Perceptron(object):
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = np.zeros((input_num, 1))
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        return self.activator(input_vec.dot(self.weights) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            print(i)
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        for i in range(len(input_vecs)):
            output = self.predict(input_vecs[i, :])
            self._update_weights(input_vecs[i, :], output, labels[i, :], rate)

    def _update_weights(self, input_vec, output, label, rate):
        delta = label - output
        print(delta[0])
        self.weights = self.weights + rate * delta * np.reshape(input_vec, (len(input_vec), 1))
        self.bias += rate * delta


def relu(x):
    return x if x > 0 else 0


def get_training_dataset():
    raw_vecs = [[1,1], [0,0], [1,0], [0,1]]
    size_vecs = len(raw_vecs)
    input_vecs = np.reshape(raw_vecs, (size_vecs, 2))

    raw_labels = [1,0,0,0]
    size_labels = len(raw_labels)
    labels = np.reshape(raw_labels, (size_labels, 1))
    return input_vecs, labels


def train_and_perceptron():
    p = Perceptron(2, relu)
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 1000, 0.1)
    return p

if __name__ == '__main__':
    and_perception = train_and_perceptron()
    print(and_perception.predict(np.reshape([1, 1], (1, 2)))[0][0])
    print('0 and 0 = %d' % and_perception.predict(np.reshape([1, 0], (1, 2)))[0][0])
    print('1 and 0 = %d' % and_perception.predict(np.reshape([0, 0], (1, 2)))[0][0])
    print('0 and 1 = %d' % and_perception.predict(np.reshape([0, 1], (1, 2)))[0][0])
