#import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import copy
import itertools
import cma
#from profilestats import profile
import numpy as np
#import numpy as gpu
import gnumpy as gpu
import functools

#has_cuda = False

gpu.vstack = gpu.concatenate
gpu.hstack = functools.partial(gpu.concatenate, axis=1)
gpu.arctanh = lambda x: .5 * gpu.log((1 + x) / (1 - x))

np.rand = np.random.rand
np.garray = np.array

class EchoStateNetwork(object):

    def __init__(self,
                 n_input_units,
                 width,
                 height,
                 n_output_units,
                 connectivity,
                 input_scaling,
                 input_shift,
                 teacher_scaling,
                 teacher_shift,
                 noise_level,
                 spectral_radius,
                 feedback_scaling,
                 leakage=0,
                 time_constants=None,
                 reservoir_activation_function='tanh',
                 output_activation_function='identity'):

        self.n_input_units = n_input_units
        self.width = width
        self.height = height
        self.n_internal_units = width * height
        self.n_output_units = n_output_units
        self.connectivity = connectivity
        self.input_scaling = input_scaling
        self.input_shift = input_shift
        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift
        self.noise_level = noise_level
        self.spectral_radius = spectral_radius
        self.feedback_scaling = feedback_scaling
        self.leakage = leakage
#        if time_constants is None:
#            time_constants = [1] * n_internal_units
        self.time_constants = time_constants
        self.reservoir_activation_function = function_from_name(reservoir_activation_function) # maybe cuda
        self.callback = None
        self.callback_every = None
        self.output_activation_function, self.inverse_output_activation_function = function_from_name(output_activation_function, return_inverse=True)

        self.reset()

    def reset(self):
        self.input_weights = self._generate_input_weights()
        self.internal_weights = self._generate_internal_weights()
        self.feedback_weights = self._generate_feedback_weights()
        self.output_weights = gpu.zeros((
            self.n_output_units, self.n_internal_units + self.n_input_units))

        scaled_feedback_weights = gpu.dot(self.feedback_weights, gpu.diagflat(self.feedback_scaling))
        scaled_feedback_weights = scaled_feedback_weights.reshape((len(self.feedback_weights), self.n_output_units))
        self.fixed_weights = gpu.hstack((self.internal_weights, self.input_weights, scaled_feedback_weights))

        self.reset_state()

    def reset_state(self):
        self.total_state = gpu.zeros((self.n_input_units + self.n_internal_units +
                                      self.n_output_units, 1))
        self.internal_state = gpu.zeros((self.n_internal_units, 1))
        self.internal_state = gpu.zeros((self.n_internal_units, 1))

    def serialize(self):
        return {
            'input_weights': self.input_weights,
            'internal_weights': self.internal_weights,
            'feedback_weights': self.feedback_weights,
            'output_weights': self.output_weights,
        }

    def unserialize(self, obj):
        self.input_weights = obj['input_weights']
        self.internal_weights = obj['internal_weights']
        self.feedback_weights = obj['feedback_weights']
        self.output_weights = obj['output_weights']

    #@profile
    def train(self, input, output, n_forget_points=0, reset_points=None):
        assert len(input) == len(output)
        assert input.shape[1] == self.n_input_units
        assert output.shape[1] == self.n_output_units

        state_matrix = self._compute_state_matrix(input, output, n_forget_points, reset_points=reset_points)
        teacher_matrix = self._compute_teacher_matrix(output, n_forget_points)
        self.output_weights = linear_regression(state_matrix, teacher_matrix)

        return state_matrix

    def test(self, input, n_forget_points=0, reset_points=None, actual_output=None):
        state_matrix = self._compute_state_matrix(input, n_forget_points=n_forget_points,
                                                  reset_points=reset_points, actual_output=actual_output)
        output = gpu.dot(state_matrix, self.output_weights.T)
        output = self.output_activation_function(output)
        output -= self.teacher_shift
        output /= self.teacher_scaling

        return output

    def _compute_state_matrix(self, input, output=None, n_forget_points=0,
                              reset_points=None, actual_output=None):
        state_matrix = gpu.zeros((len(input) - n_forget_points,
                                  self.n_input_units + self.n_internal_units))

        if self.callback:
            callback_state = gpu.zeros((self.callback_every, self.n_input_units + self.n_internal_units + self.n_output_units))

        if reset_points is not None:
            reset_points = set(reset_points)

        for i, input_point in enumerate(input):

            if reset_points is not None and i in reset_points:
                self.reset_state()

            scaled_input = (self.input_scaling * input_point + self.input_shift).reshape(
                len(input_point), 1)

            self.total_state[self.n_internal_units :
                             self.n_internal_units + self.n_input_units] = scaled_input
            self._update_internal_state()

            self.total_state[:self.n_internal_units, :] = self.internal_state
            self.total_state[self.n_internal_units:self.n_internal_units + self.n_input_units, :] = scaled_input

            if output is None:
                scaled_output = self.output_activation_function(self.output_weights.dot(self.total_state[:-self.n_output_units, :]))
            else:
                scaled_output = self.teacher_scaling * output[i,:] + self.teacher_shift
                scaled_output = scaled_output.reshape((len(scaled_output), 1))

            self.total_state[-self.n_output_units:, :] = scaled_output

            if i >= n_forget_points:
                state_matrix[i - n_forget_points, :self.n_internal_units] = self.internal_state.T
                state_matrix[i - n_forget_points, self.n_internal_units:] = scaled_input.T

            if i % 100 == 0:
                print i, len(input)

            if self.callback:
                callback_state[i % self.callback_every,:] = gpu.vstack((scaled_input, self.internal_state, scaled_output)).T
                if (i + 1) % self.callback_every == 0:
                    print i, len(input)
                    if actual_output is not None:
                        self.callback(callback_state, actual_output[max(0, i - self.callback_every):i, :])
                    else:
                        self.callback(callback_state)

        return state_matrix

    def _update_internal_state(self):
        self.internal_state = self.reservoir_activation_function(gpu.dot(self.fixed_weights, self.total_state))
        self.internal_state += self.noise_level * (gpu.rand(self.n_internal_units, 1) - .5)

    def _compute_teacher_matrix(self, output, n_forget_points):
        teacher = self.teacher_scaling * output[n_forget_points:, :] + self.teacher_shift
        return self.inverse_output_activation_function(teacher)

    def _generate_input_weights(self):
        return 2 * gpu.rand(self.n_internal_units, self.n_input_units) - 1

    def _generate_internal_weights(self):
        internal_weights = scipy.sparse.rand(
            self.n_internal_units, self.n_internal_units, self.connectivity)
        internal_weights = gpu.garray(internal_weights.todense())
        return self._normalise_internal_weights(internal_weights)

    def _normalise_internal_weights(self, internal_weights):
        internal_weights[gpu.where(internal_weights != 0)] -= .5
        if gpu == np:
            eigvals = np.linalg.eigvals(internal_weights)
        else:
            eigvals = np.linalg.eigvals(internal_weights.as_numpy_array())
        radius = gpu.max(gpu.abs(eigvals))
        internal_weights /= radius
        internal_weights *= self.spectral_radius
        return internal_weights

    def _generate_feedback_weights(self):
        return 2 * gpu.rand(self.n_internal_units, self.n_output_units) - 1

    def get_input_weight(self, i, x2, y2):
        return self.input_weights[self.point_to_index(x2, y2), i]

    def get_internal_weight(self, x1, y1, x2, y2):
        return self.internal_weights[self.point_to_index(x2, y2), self.point_to_index(x1, y1)]

    def get_feedback_weight(self, i, x2, y2):
        return self.feedback_weights[self.point_to_index(x2, y2), i]

    def get_internal_to_output_weight(self, i, x2, y2):
        return self.output_weights[i, self.point_to_index(x2, y2)]

    def get_input_to_output_weight(self, i, j):
        return self.output_weights[i, self.n_internal_units + j]

    def point_to_index(self, x, y):
        return x % self.width + y * self.width

class NeighbourESN(EchoStateNetwork):

    def __init__(self,
                 n_input_units,
                 width,
                 height,
                 n_output_units,
                 input_scaling,
                 input_shift,
                 teacher_scaling,
                 teacher_shift,
                 noise_level,
                 spectral_radius,
                 feedback_scaling,
                 leakage=0,
                 time_constants=None,
                 reservoir_activation_function=np.tanh,
                 output_activation_function='identity'):

        super(NeighbourESN, self).__init__(
            n_input_units=n_input_units,
            width=width,
            height=height,
            n_output_units=n_output_units,
            input_scaling=input_scaling,
            input_shift=input_shift,
            teacher_scaling=teacher_scaling,
            teacher_shift=teacher_shift,
            noise_level=noise_level,
            spectral_radius=spectral_radius,
            feedback_scaling=feedback_scaling,
            leakage=leakage,
            time_constants=time_constants,
            reservoir_activation_function=reservoir_activation_function,
            output_activation_function=output_activation_function,
            connectivity=None
        )

    # def _generate_input_weights(self):
    #     weights = np.zeros((self.n_internal_units, self.n_input_units))
    #     top_row = 2 * np.random.random((self.width, self.n_input_units)) - 1
    #     weights[0:self.width, :] = top_row
    #     return weights

    # def _generate_feedback_weights(self):
    #     weights = np.zeros((self.n_internal_units, self.n_output_units))
    #     bottom_row = 2 * np.random.random((self.height, self.n_output_units)) - 1
    #     weights[self.width * (self.height - 1):self.n_internal_units, :] = bottom_row
    #     return weights

    def _generate_internal_weights(self):
        weights = np.zeros((self.n_internal_units, self.n_internal_units))

        for x in xrange(self.width - 1):
            for y in xrange(self.height - 1):
                p1 = (x, y)
                p2 = (x + 1, y)
                p3 = (x, y + 1)
                p4 = (x + 1, y + 1)
                neighbours = [(p1, p2), (p1, p3), (p1, p4), (p3, p2)]
                if x == self.width - 2:
                    neighbours.append((p2, p4))
                if y == self.height - 2:
                    neighbours.append((p3, p4))
                for start, end in neighbours:
                    if np.random.rand() < .5:
                        start, end = end, start
                    weights[self.point_to_index(*start),
                            self.point_to_index(*end)] = np.random.rand()

        return self._normalise_internal_weights(weights)


def nrmse(estimated, correct):
    if gpu == np:
        correct_variance = np.var(correct)
    else:
        correct_variance = np.var(correct.as_numpy_array())
    if correct_variance == 0:
        correct_variance = 0.01 # hack

    return gpu.sqrt(mean_error(estimated, correct) / correct_variance)

def mean_error(estimated, correct):
    n_forget_points = len(correct) - len(estimated)
    correct = correct[n_forget_points:, :]
    return sum((estimated - correct) ** 2) / len(estimated)

class Optimiser(object):

    def __init__(self, esn, input, output, forget_points=0, test=nrmse):
        self.esn = esn
        self.input = input
        self.output = output
        self.forget_points = forget_points
        self.test = test

        self.input_ndx = esn.input_weights != 0
        self.internal_ndx = esn.internal_weights != 0
        self.feedback_ndx = esn.feedback_weights != 0

    def initial_params(self):
        return (list(self.esn.input_weights[self.input_ndx].flatten()) +
                list(self.esn.internal_weights[self.internal_ndx].flatten()) +
                list(self.esn.feedback_weights[self.feedback_ndx].flatten()))

    def evaluate(self, params):
        self.esn.reset_state()

        self.esn.input_weights[self.input_ndx] = params[:np.sum(self.input_ndx)]
        self.esn.internal_weights[self.internal_ndx] = params[np.sum(self.input_ndx):np.sum(self.input_ndx) + np.sum(self.internal_ndx)]
        self.esn.feedback_weights[self.feedback_ndx] = params[np.sum(self.input_ndx) + np.sum(self.internal_ndx):]

        self.esn.train(self.input, self.output, self.forget_points)
        estimated_output = self.esn.test(self.input)
        error = np.sum(self.test(estimated_output, self.output))

        print error
        return error

def function_from_name(name, return_inverse=False):
    func = None
    inverse = None
    if name == 'identity':
        func = lambda x: x
        inverse = lambda x: x
    elif name == 'tanh':
        func = gpu.tanh
        inverse = gpu.arctanh
    else:
        raise Exception('Unknown function: %s' % name)

    if return_inverse:
        return func, inverse
    else:
        return func

def linear_regression(state_matrix, teacher_matrix):
    # use actual gpu linalg packages here instead
    # return cuda_linalg.transpose(maybe_cuda_dot(cuda_linalg.pinv(maybe_cuda(state_matrix)), teacher_matrix))
    run_length = state_matrix.shape[0]
    cov_mat = state_matrix.T.dot(state_matrix / run_length)
    p_vec = state_matrix.T.dot(teacher_matrix / run_length)
    if gpu == np:
        inv = gpu.garray(np.linalg.inv(cov_mat))
    else:
        inv = gpu.garray(np.linalg.inv(cov_mat.as_numpy_array()))
    return (inv.dot(p_vec)).T
