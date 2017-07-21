import numpy as np

class Perceptron(object):
    def __init__(self, activator, input_num):
        self.activator = activator
        self.weights = np.zeros((input_num, 1))
        self.bias = 0

    def __str__(self):
        return 'weight:\t%s\nbias:\t%f' % (self.weights, self.bias)

    def predict(self, input_arr):
        return self.activator(input_arr.dot(self.weights) + self.bias)

    def train(self, input_arr, labels, iteration, learning_rate):
        for i in range(iteration):
            self._one_iteration(input_arr, labels, learning_rate)

    def _one_iteration(self, input_arr, labels, learning_rate):
        for i in range(len(input_arr)):
            output = self.predict(input_arr[i, :])
            self._update_weight(input_arr[i, :], output, labels[i, :], learning_rate)

    def _update_weight(self, input_arr, output, label, rate):
        delta = label - output
        self.weights = self.weights + rate * delta * np.reshape(input_arr, (len(input_arr), 1))
        self.bias = self.bias + rate * delta

    def print_weight(self):
        print(self.weights)

def f(x):
    return 1 if x > 1 else 0

def get_training_dataset():

    raw_vecs = [[1,1], [0,0], [1,0], [0,1]]
    size_vecs = len(raw_vecs)
    input_vecs = np.reshape(raw_vecs, (size_vecs, 2))

    raw_labels = [1, 0, 0, 0]
    size_labels = len(raw_labels)
    labels = np.reshape(raw_labels, (size_labels, 1))
    return input_vecs, labels

def train_and_perceptron():
    p = Perceptron(f, 2)
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p

if __name__ == '__main__':
    and_perception = train_and_perceptron()
    print(and_perception.predict(np.reshape([1, 0], (1, 2))))
    print(and_perception.print_weight())
