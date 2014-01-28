import numpy as np
import random
import scipy.sparse
from sklearn.base import BaseEstimator

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
plt.ion()

ITERATION = 0

RAINBOW = plt.get_cmap('jet')

DIRECTIONS = [
    (0, -1), (1, -1), (1, 0), (1, 1),
    (0, 1), (-1, 1), (-1, 0), (-1, -1)
]

ACTUAL_DIRECTIONS = {d: d / np.linalg.norm(d) for d in DIRECTIONS}

def opposite_direction(direction):
    x, y = direction
    return -x, -y

def random_direction():
    r = np.random.rand() * 2 * np.pi
    return np.array([np.sin(r), np.cos(r)])

class Neuron(object):

    def __init__(self, initial_direction, fixed_direction=False, sharpness=4.0, damping=0.4):
        self.neighbours = {}
        self.activation = 0
        self.direction = initial_direction
        self.fixed_direction = fixed_direction
        self.incoming_activations = {}
        self.sharpness = sharpness
        self.damping = damping

    def set_neighbour(self, direction, neuron):
        self.neighbours[direction] = neuron

    def get_outgoing_activations(self):
        outgoing_activations = {}
        for direction in self.neighbours:
            outgoing_activations[direction] = np.exp(
                -self.sharpness * np.linalg.norm(self.direction - ACTUAL_DIRECTIONS[direction])
            )

        normalisation = self.activation * self.damping / sum(outgoing_activations.values())
        for direction in self.neighbours:
            outgoing_activations[direction] = normalisation * outgoing_activations[direction]

        return outgoing_activations

    def push_activations(self):
        outgoing_activations = self.get_outgoing_activations()
        for direction, neighbour in self.neighbours.items():
            neighbour = self.neighbours[direction]
            opposite = opposite_direction(direction)
            neighbour.incoming_activations[opposite] = outgoing_activations[direction]

    def update(self, input, learning_rate):
        self.activation = input
        incoming_direction = np.array([0, 0], dtype=float)
        for direction in DIRECTIONS:
            if direction in self.incoming_activations:
                self.activation += self.incoming_activations[direction]
                incoming_direction += ACTUAL_DIRECTIONS[direction] * self.incoming_activations[direction]
        self.incoming_activations = {}

        if not self.fixed_direction:
            self.direction += learning_rate * (opposite_direction(incoming_direction) - self.direction)

        if self.activation > 4:
            self.activation = 4
        return self.activation

    def activation_towards(self, direction):
        return 

class Network(BaseEstimator):

    def __init__(self, n_inputs, width, height, spectral_radius=0.8, sharpness=4.0, damping=0.4, leakage=0.2, beta=0):
        self.n_inputs = n_inputs
        self.width = width
        self.height = height
        self.n_internal = width * height
        self.neurons = []
        self.input_weights = np.random.rand(n_inputs, self.n_internal)
        #self.spectral_radius = 1.1
        self.spectral_radius = spectral_radius
        self.leakage = leakage
        self.beta = beta

        for row in range(self.n_internal):
            self.neurons.append(Neuron(random_direction(),
                                       sharpness=sharpness, damping=damping))
        for i in range(max(1, int(np.sqrt(self.n_internal) / 6))):
            self.neurons[np.random.randint(self.n_internal)].fixed_direction = True

        for y in range(self.height):
            for x in range(self.width):
                for direction in DIRECTIONS:
                    dx, dy = direction
                    #index = self.neuron_index(row + y, column + x)
                    index = self.neuron_index((x + dx) % self.width, (y + dy) % self.height)
                    if True or( index >= 0 and index < self.n_internal):
                        neighbour = self.neurons[index]
                    else:
                        neighbour = None
                    self.neurons[self.neuron_index(x, y)].set_neighbour(direction, neighbour)

    def neuron_index(self, x, y):
        return x + y * self.height

    def neuron_position(self, index):
        return index % self.width, index // self.width

    def set_random_internal_weights(self, connectivity):
        weights = scipy.sparse.rand(
            self.n_internal, self.n_internal, connectivity).todense()
        self.internal_weights = self.normalise_weights(weights)

    def pretrain(self, inputs, split_points=None):
        activations = np.zeros((self.n_internal, 1))
        history = np.zeros((len(inputs), self.n_internal))

        learning_rate = .4
        inputs = inputs.copy()
        np.random.shuffle(inputs)

        for i, x in enumerate(inputs):
            x = x[np.newaxis].T

            if False and i % 1 == 0:
                print '%d/%d' % (i, len(inputs))

            if split_points is not None and i in split_points:
                activations = np.zeros((self.n_internal, 1))

            activations = self.update_activations(x, learning_rate)
            history[i, :] = activations

            learning_rate *= .96
            #print learning_rate

            #print np.max(activations), np.min(activations), learning_rate

            #if True or i % 5 == 0:
            #    self.plot_arrows()

            if i > 10:
                break

        self.internal_weights = self.to_weight_matrix()

    def update_activations(self, x, learning_rate):
        activations = []
        for neuron in self.neurons:
            neuron.push_activations()
        for i, neuron in enumerate(self.neurons):
            if True or i == 0:
                #activations.append(neuron.update(1, learning_rate))
                activations.append(neuron.update(x.T.dot(self.input_weights[:, i]), learning_rate))

            else:
                activations.append(neuron.update(0, learning_rate))

        return activations

    def to_weight_matrix(self):
        weights = np.zeros((self.n_internal, self.n_internal))
        for i, neuron in enumerate(self.neurons):
            x, y = self.neuron_position(i)
            outgoing_activations = neuron.get_outgoing_activations()
            for direction in neuron.neighbours:
                nx, ny = np.array([x, y]) + direction
                nx = nx % self.width
                ny = ny % self.height
                neighbour_index = self.neuron_index(nx, ny)
                weight = outgoing_activations[direction]
#                if np.random.rand() > .5:
#                    weight *= -1
                weights[i, neighbour_index] = weight

        weights = self.normalise_weights(weights)
        return weights

    def normalise_weights(self, weights):
        eigs = np.abs(np.linalg.eigvals(weights))
        weights /= np.max(eigs)
        weights *= self.spectral_radius
        return weights

    def plot_arrows(self):
        global ITERATION

        plt.clf()

        for i, neuron in enumerate(self.neurons):
            x, y = self.neuron_position(i)
            dx, dy = neuron.direction
            arrow = mpatches.Arrow((x + .5) / self.width, 1 - (y + .5) / self.height,
                                   dx / self.width, - dy / self.height, width=.01,
                                   color=('r' if neuron.fixed_direction else 'k'))
            plt.gca().add_patch(arrow)

        plt.draw()
        #plt.savefig('figs/%04d.png' % ITERATION)

        ITERATION += 1


    def plot_state(self, i, history):
        plt.clf()
        ax = plt.gca()
        w = float(self.width + 1)
        h = float(self.height + 1)

        minw = np.min(self.internal_weights)
        maxw = np.max(self.internal_weights)
        diffw = maxw - minw

        for n, neuron in enumerate(self.neurons):
            x, y = self.neuron_position(n)
            line_y = -history[i - 100:i:4, n]
            #line_y = history[max(i - 20, 0):i, n] + .25
            line_x = np.arange(len(line_y)) / 50.0
            #line_x = np.arange(len(line_y)) / 40.0
            line = mlines.Line2D((line_x + x + .75) / w, 1 - (line_y + y + 1) / h)
            ax.add_line(line)

            k = .2
            for dx, dy in DIRECTIONS:
                weight = self.internal_weights[self.neuron_index(x, y), self.neuron_index((x + dx) % self.width, (y + dy) % self.height)]

                plt.arrow(
                    (1 + x + (1 - k) / 2 * dx) / w,
                    1 - (1 + y + (1 - k) / 2 * dy) / h,
                    (k * dx) / w,
                    -(k * dy) / h,
                    width=.0002,
                    color=str(weight / maxw),
                )

        plt.draw()

    def fit(self, inputs, outputs, split_points=None):
        self.input_weights -= .5
        self.input_weights *= 1.5
        print np.min(self.input_weights), np.max(self.input_weights)

#        self.input_weights[:, :] = 0
#        for i in range(self.n_inputs):
#            for j in range(5):
#                self.input_weights[i, np.random.randint(self.n_internal)] = np.sign(np.random.rand() - .5)

        history = self.get_history(inputs, split_points)
        self.output_weights = self.linear_regression(history, outputs)

    def get_history(self, inputs, split_points):
        activations = np.zeros((self.n_internal, 1))
        history = np.zeros((len(inputs), self.n_internal))

        fixed_weights = np.vstack((self.input_weights, self.internal_weights))

        for i, x in enumerate(inputs):
            x = x[np.newaxis].T * .3

            if False and i % 10000 == 0:
                print 'train %d/%d' % (i, len(inputs))
                #self.plot_state(i, history)

            if split_points is not None and i in split_points:
                activations = np.zeros((self.n_internal, 1))

            total_state = np.vstack((x, activations))
            activations = self.leakage * np.tanh(fixed_weights.T.dot(total_state)) + (1 - self.leakage) * activations

            history[i, :] = activations.T

        return history

    def predict(self, inputs, split_points=None):
        history = self.get_history(inputs, split_points)
        return history.dot(self.output_weights)

    def linear_regression(self, history, outputs):
        #return outputs.T.dot(history).dot(np.linalg.inv(history.T.dot(history) + beta)).T
        return np.linalg.inv(
            history.T.dot(history) + self.beta * np.eye(history.shape[1])
        ).dot(history.T).dot(outputs)
