import numpy as np
import math


def sigmod(x):
    return 1 / (1 + math.e ^ x)


class Node(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        self.output = output

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        self.upstream.append(conn)

    def calc_output(self):
        conn = {}
        conn.output = []
        conn.weight = []
        for i in range(len(self.upstream)):
            conn.output.append(self.upstream[i].upstream_node.output)
            conn.weight.append(self.upstream[i].weight)
        output = sigmod(np.reshape(conn.output, (1, len(conn.output))).dot(np.reshape(conn.weight, (len(conn.weight), 1))))


