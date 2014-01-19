import numpy as np
import random
import scipy.sparse
from sklearn.base import BaseEstimator

import matplotlib.pyplot as plt
plt.ion()

DIRECTIONS = [
    (0, -1), (1, -1), (1, 0), (1, 1),
    (0, 1), (-1, 1), (-1, 0), (-1, -1)
]

def opposite_direction(direction):
    x, y = direction
    return -x, -y

def random_direction():
    return np.array(random.choice(DIRECTIONS), dtype=float)

class Neuron(object):

    def __init__(self):
        self.neighbours = {}
        self.activation = 0
        self.direction = random_direction()
        self.incoming_activations = {}

    def set_neighbour(self, direction, neuron):
        self.neighbours[direction] = neuron

    def push_activations(self):
        for direction in DIRECTIONS:
            neighbour = self.neighbours[direction]
            neighbour.incoming_activations[opposite_direction(direction)] = self.activation_towards(direction)

    def update(self, input, learning_rate):
        self.activation = input
        incoming_direction = np.array([0, 0], dtype=float)
        for direction in DIRECTIONS:
            self.activation += self.incoming_activations[direction]
            incoming_direction += np.array(direction) * self.incoming_activations[direction]

        self.activation = np.tanh(self.activation)
        self.direction += learning_rate * (incoming_direction - self.direction)

        return self.activation

    def activation_towards(self, direction):
        return max(0, min(1, (1 - np.linalg.norm(self.direction - direction) / 2.3))) * self.activation

class Network(BaseEstimator):

    def __init__(self, n_inputs, width, height):
        self.n_inputs = n_inputs
        self.width = width
        self.height = height
        self.n_internal = width * height
        self.neurons = []
        self.input_weights = np.random.rand(n_inputs, self.n_internal) - .5

        for row in range(self.n_internal):
            self.neurons.append(Neuron())
        for row in range(self.height):
            for column in range(self.width):
                for direction in DIRECTIONS:
                    x, y = direction
                    self.neurons[self.neuron_index(row, column)].set_neighbour(
                        direction, self.neurons[self.neuron_index(
                        (row + y) % self.height, (column + x) % self.width)])

    def neuron_index(self, row, column):
        return row * self.width + column

    def pretrain(self, inputs, split_points=None):
        activations = np.zeros((self.n_internal, 1))
        history = np.zeros((len(inputs), self.n_internal))

        learning_rate = 1

        for i, x in enumerate(inputs):
            x = x[np.newaxis].T

            if i % 1 == 0:
                print '%d/%d' % (i, len(inputs))

            if split_points is not None and i in split_points:
                activations = np.zeros((self.n_internal, 1))

            activations = self.update_activations(x, learning_rate)
            history[i, :] = activations

            learning_rate *= .98
            #print learning_rate

    def update_activations(self, x, learning_rate):
        activations = []
        for neuron in self.neurons:
            neuron.push_activations()
        for i, neuron in enumerate(self.neurons):
            activations.append(neuron.update(x.T.dot(self.input_weights[:, i]), learning_rate))
        directions = []

        import math
        for i, neuron in enumerate(self.neurons):
            x, y = neuron.direction
            directions.append(math.atan2(x, y))
            #directions.append(np.linalg.norm(neuron.direction))
            #directions.append(neuron.activation)

        #plt.clf()
        #plt.imshow(np.array(directions).reshape((self.height, self.width)), interpolation='none')
        #plt.draw()

        #print np.min(directions), np.max(directions)

        return activations

    def linear_regression(self, history, outputs, beta=0):
        return outputs.T.dot(history).dot(np.linalg.inv(history.T.dot(history) + beta)).T

