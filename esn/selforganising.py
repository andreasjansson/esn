import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import matplotlib.pyplot as plt
plt.ion()

class SelfOrganisingReservoir(BaseEstimator):

    def __init__(self, n_inputs, n_outputs,
                 width, height,
                 input_scaling=30, internal_scaling=2,
                 leakage=.5,
                 learning_rate_start=0.04, learning_rate_end=0.01,
                 neighbourhood_width_start=2, neighbourhood_width_end=0.001):

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.width = width
        self.height = height
        self.input_scaling = input_scaling
        self.internal_scaling = internal_scaling
        self.leakage = leakage
        self.n_internal = width * height
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
        activations = np.zeros((self.n_internal, 1))
        history = np.zeros((len(inputs), self.n_internal))

        for i, x in enumerate(inputs):
            x = x[np.newaxis].T

            if split_points is not None and i in split_points:
                activations = np.zeros((self.n_internal, 1))

            activations = self.update_activations(x, activations)
            history[i, :] = activations.T

        self.output_weights = self.linear_regression(history, outputs)

        return self

    def predict(self, inputs, split_points=None):
        activations = np.zeros((self.n_internal, 1))
        outputs = np.zeros((len(inputs), self.n_outputs))

        for i, x in enumerate(inputs):
            x = x[np.newaxis].T

            if split_points is not None and i in split_points:
                activations = np.zeros((self.n_internal, 1))

            activations = self.update_activations(x, activations)

            output = activations.T.dot(self.output_weights)
            outputs[i, :] = output

        return outputs

    def linear_regression(self, history, outputs, beta=0):
        #return np.linalg.pinv(history).dot(outputs)

        return outputs.T.dot(history).dot(np.linalg.inv(history.T.dot(history) + beta)).T

    def update_weights(self, target, values, learning_rate, neighbourhood_width):
        best_unit = np.argmin(np.linalg.norm(values - target, axis=0))
        x = best_unit % self.width
        y = best_unit // self.width

        dist_x = np.abs(np.arange(-x, self.width - x))
        dist_y = np.abs(np.arange(-y, self.height - y))
        distances = (dist_x + dist_y[np.newaxis].T).ravel()
        #distances = np.abs(best_unit - np.arange(self.n_internal))

        distribution = np.exp(- (distances ** 2) / (neighbourhood_width ** 2))

        for i, v in enumerate(values.T):
            values[:, i] += (learning_rate * distribution[i] * (target.T - v)).ravel()

        #values += (learning_rate * distribution)[np.newaxis].dot((target - values).T)

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
