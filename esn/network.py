import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import copy
import itertools
import cma
#from profilestats import profile
import functools

class EchoStateNetwork(object):

    def __init__(self,
                 n_input_units,
                 width,
                 height,
                 n_output_units,
                 connectivity,
                 input_scaling,
                 input_shift,
                 spectral_radius,
                 teacher_scaling=None,
                 teacher_shift=None,
                 noise_level=0,
                 feedback_scaling=0,
                 leakage=0,
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
        self.reservoir_activation_function = function_from_name(reservoir_activation_function) # maybe cuda
        self.callback = None
        self.callback_every = None
        self.output_activation_function, self.inverse_output_activation_function = function_from_name(output_activation_function, return_inverse=True)

        self.reset()

    def reset(self):
        self.input_weights = self._generate_input_weights()
        self.internal_weights = self._generate_internal_weights()
        self.feedback_weights = self._generate_feedback_weights()
        self.output_weights = np.zeros((
            self.n_output_units, self.n_internal_units + self.n_input_units), dtype='float32')

        scaled_feedback_weights = np.dot(self.feedback_weights, np.diagflat(self.feedback_scaling))
        scaled_feedback_weights = scaled_feedback_weights.reshape((len(self.feedback_weights), self.n_output_units))
        self.fixed_weights = scipy.sparse.hstack((
                self.internal_weights,
                scipy.sparse.csr_matrix(self.input_weights),
                scipy.sparse.csr_matrix(scaled_feedback_weights)), dtype='float32').tocsr()

        self.reset_state()

    def reset_state(self):
        self.total_state = np.zeros(self.n_input_units + self.n_internal_units +
                                    self.n_output_units, dtype='float32')
        self.internal_state = np.zeros(self.n_internal_units, dtype='float32')

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
        output = np.dot(state_matrix, self.output_weights.T)
        output = self.output_activation_function(output)
        if self.teacher_shift is not None:
            output -= self.teacher_shift
        if self.teacher_scaling is not None:
            output /= self.teacher_scaling

        return output

    def _compute_state_matrix(self, input, output=None, n_forget_points=0,
                              reset_points=None, actual_output=None):
        state_matrix = np.zeros((len(input) - n_forget_points,
                                  self.n_input_units + self.n_internal_units), dtype='float32')

        if self.callback:
            callback_state = np.zeros((self.callback_every, self.n_input_units + self.n_internal_units + self.n_output_units), dtype='float32')

        if reset_points is not None:
            reset_points = set(reset_points)

        for i, input_point in enumerate(input):

            if reset_points is not None and i in reset_points:
                self.reset_state()

            scaled_input = (self.input_scaling * input_point + self.input_shift)

            self.total_state[self.n_internal_units :
                             self.n_internal_units + self.n_input_units] = scaled_input
            if self.leakage:
                self._update_internal_state_leaky()
            else:
                self._update_internal_state()

            self.total_state[:self.n_internal_units] = self.internal_state
            self.total_state[self.n_internal_units:self.n_internal_units + self.n_input_units] = scaled_input

            if output is None:
                scaled_output = self.output_activation_function(self.output_weights.dot(self.total_state[:-self.n_output_units]))
            else:
                scaled_output = output[i,:]
                if self.teacher_scaling is not None:
                    scaled_output *= self.teacher_scaling
                if self.teacher_shift is not None:
                    scaled_output += self.teacher_shift

            self.total_state[-self.n_output_units:] = scaled_output

            if i >= n_forget_points:
                state_matrix[i - n_forget_points, :self.n_internal_units] = self.internal_state
                state_matrix[i - n_forget_points, self.n_internal_units:] = scaled_input

            if i % 1000 == 0:
                print i, len(input)

            if self.callback:
                callback_state[i % self.callback_every,:] = np.vstack((scaled_input, self.internal_state, scaled_output)).T
                if (i + 1) % self.callback_every == 0:
                    print i, len(input)
                    if actual_output is not None:
                        self.callback(callback_state, actual_output[max(0, i - self.callback_every):i, :])
                    else:
                        self.callback(callback_state)

        return state_matrix

    def _update_internal_state(self):
        self.internal_state = self.reservoir_activation_function(self.fixed_weights.dot(self.total_state))
        self.internal_state += self.noise_level * (np.random.rand(self.n_internal_units) - .5).astype('float32')


    def _update_internal_state_leaky(self):
        previous_internal_state = self.total_state[0:self.n_internal_units]
        self.internal_state = ((1 - self.leakage) * previous_internal_state +
                               self.leakage * self.reservoir_activation_function(
                                   self.fixed_weights.dot(self.total_state)))
        self.internal_state += self.noise_level * (np.random.rand(self.n_internal_units) - .5).astype('float32')

    def _compute_teacher_matrix(self, output, n_forget_points):
        teacher = output[n_forget_points:, :]
        if self.teacher_scaling is not None:
            teacher *= self.teacher_scaling
        if self.teacher_shift is not None:
            teacher += self.teacher_shift

        return self.inverse_output_activation_function(teacher)

    def _generate_input_weights(self):
        return 2 * np.random.rand(self.n_internal_units, self.n_input_units).astype('float32') - 1

    def _generate_internal_weights(self):
        internal_weights = scipy.sparse.rand(
            self.n_internal_units, self.n_internal_units, self.connectivity).tocsr()
        return self._normalise_internal_weights(internal_weights)

    def _normalise_internal_weights(self, internal_weights):
        internal_weights.data -= .5
        attempts = 5
        for i in range(attempts):
            try:
                eigvals = scipy.sparse.linalg.eigs(
                    internal_weights, k=1, which='LM',
                    return_eigenvectors=False, tol=.02, maxiter=5000)
                break
            except scipy.sparse.linalg.ArpackNoConvergence:
                pass
        else:
            print 'scipy.sparse.linalg failed to converge, falling back to numpy.linalg'
            eigvals = np.linalg.eigvals(internal_weights.todense())
        radius = np.abs(np.max(eigvals))
        internal_weights /= radius
        internal_weights *= self.spectral_radius
        return internal_weights

    def _generate_feedback_weights(self):
        return 2 * np.random.rand(self.n_internal_units, self.n_output_units).astype('float32') - 1

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
        weights = np.zeros((self.n_internal_units, self.n_internal_units), dtype='float32')

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
    correct_variance = np.var(correct)
    if correct_variance == 0:
        correct_variance = 0.01 # hack

    return np.sqrt(mean_error(estimated, correct) / correct_variance)

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
        func = np.tanh
        inverse = np.arctanh
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
    inv = np.array(np.linalg.inv(cov_mat), dtype='float32')
    return (inv.dot(p_vec)).T
