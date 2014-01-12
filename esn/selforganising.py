import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import matplotlib.pyplot as plt
plt.ion()

class SelfOrganisingReservoir(BaseEstimator):

    def __init__(self,
                 n_inputs, n_outputs,
                 width, height,
                 input_scaling=30, internal_scaling=2,
                 leakage=.5,
                 learning_rate_start=0.04, learning_rate_end=0.01,
                 neighbourhood_width_start=2, neighbourhood_width_end=0.001):

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.width = width
        self.height = height
        self.n_internal = width * height
        self.input_scaling = input_scaling / float(n_inputs)
        self.internal_scaling = internal_scaling / float(self.n_internal)
        self.leakage = leakage
        self.learning_rate_start = learning_rate_start
        self.learning_rate_end = learning_rate_end
        self.neighbourhood_width_start = neighbourhood_width_start
        self.neighbourhood_width_end = neighbourhood_width_end

        self.input_weights = self.initial_input_weights()
        self.internal_weights = self.initial_internal_weights()
        self.output_weights = None

    def pretrain(self, inputs, split_points=None):
        learning_rates = np.linspace(
            self.learning_rate_start, self.learning_rate_end, len(inputs))
        neighbourhood_widths = np.linspace(
            self.neighbourhood_width_start, self.neighbourhood_width_end, len(inputs))

        activations = np.zeros((self.n_internal, 1))

        for i, (x, learning_rate, neighbourhood_width) in enumerate(zip(
                inputs, learning_rates, neighbourhood_widths)):
            x = x[np.newaxis].T

            if i % 1000 == 0:
                print 'pretrain %d/%d' % (i, len(inputs))

            if split_points is not None and i in split_points:
                activations = np.zeros((self.n_internal, 1))

            activations = self.update_activations(x, activations)
            self.update_weights(x, self.input_weights, learning_rate, neighbourhood_width)
            self.update_weights(activations, self.internal_weights, learning_rate, neighbourhood_width)

            #plt.clf()
            #plt.imshow(self.internal_weights, interpolation='none')
            #import ipdb; ipdb.set_trace()

            #print np.round(self.internal_weights.ravel(), 3)
        
    def fit(self, inputs, outputs, split_points=None):
        history = self.get_activation_history(inputs, split_points)
        self.output_weights = self.linear_regression(history, outputs)
        return self

    def predict(self, inputs, split_points=None):
        history = self.get_activation_history(inputs, split_points)
        outputs = history.dot(self.output_weights)
        return outputs

    def get_activation_history(self, inputs, split_points=None):
        activations = np.zeros((self.n_internal, 1))
        history = np.zeros((len(inputs), self.n_internal))

        for i, x in enumerate(inputs):
            x = x[np.newaxis].T

            if i % 1000 == 0:
                print '%d/%d' % (i, len(inputs))

            if split_points is not None and i in split_points:
                activations = np.zeros((self.n_internal, 1))

            activations = self.update_activations(x, activations)
            history[i, :] = activations.T

        return history

    def linear_regression(self, history, outputs, beta=0):
        return outputs.T.dot(history).dot(np.linalg.inv(history.T.dot(history) + beta)).T

    def update_weights(self, target, values, learning_rate, neighbourhood_width):
        best_unit = np.argmin(np.linalg.norm(values - target, axis=0))
        x = best_unit % self.width
        y = best_unit // self.width

        dist_x = np.abs(np.arange(-x, self.width - x))
        dist_y = np.abs(np.arange(-y, self.height - y))
        distances = (dist_x + dist_y[np.newaxis].T).ravel()

        distribution = np.exp(- (distances ** 2) / (neighbourhood_width ** 2))

        values += learning_rate * distribution * (target - values)

    def update_activations(self, x, activations):
        input_diff = np.linalg.norm(self.input_weights - x, axis=0) ** 2
        internal_diff = np.linalg.norm(self.internal_weights - activations, axis=0) ** 2
        new_activations = np.exp(-self.input_scaling * input_diff -
                                 self.internal_scaling * internal_diff)[np.newaxis].T
        return (1. - self.leakage) * activations + self.leakage * new_activations
                


    def initial_input_weights(self):
        a = np.linspace(1. / self.n_inputs, 1, self.n_inputs)[np.newaxis]
        b = np.linspace(1. / self.n_internal, 1, self.n_internal)[np.newaxis]
        return a.T.dot(b)
        
    def initial_internal_weights(self):
        return np.eye(self.n_internal)


class DeepSelfOrganisingReservoir(BaseEstimator):

    def __init__(self,
                 n_inputs, n_outputs,
                 sizes,
                 input_scaling=30, internal_scaling=2,
                 leakage=.5,
                 learning_rate_start=0.04, learning_rate_end=0.01,
                 neighbourhood_width_start=2, neighbourhood_width_end=0.001):

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.sizes = sizes

        self.layers = []
        for i, size in enumerate(sizes):
            self.layers.append(SelfOrganisingReservoir(
                n_inputs=n_inputs if i == 0 else sizes[i - 1][0] * sizes[i - 1][1],
                n_outputs=n_outputs if i == len(sizes) - 1 else sizes[i + 1][0] * sizes[i + 1][1],
                width=size[0],
                height=size[1],
                input_scaling=input_scaling,
                internal_scaling=internal_scaling,
                leakage=leakage,
                learning_rate_start=learning_rate_start,
                learning_rate_end=learning_rate_end,
                neighbourhood_width_start=neighbourhood_width_start,
                neighbourhood_width_end=neighbourhood_width_end,
            ))

    def pretrain(self, inputs, split_points=None):
        for layer in self.layers:
            layer.pretrain(inputs, split_points)
            inputs = layer.get_activation_history(inputs, split_points)

    def fit(self, inputs, outputs, split_points=None):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                inputs = layer.get_activation_history(inputs, split_points)
            else:
                layer.fit(inputs, outputs, split_points)

    def predict(self, inputs, split_points=None):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                inputs = layer.get_activation_history(inputs, split_points)
            else:
                return layer.predict(inputs, split_points)
