import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import copy
import itertools
import cma
#from profilestats import profile

try:
    import pycuda.driver as drv
    import pycuda.gpuarray as gpuarray
    import pycuda.cumath as cumath
    import pycuda.autoinit
    from scikits.cuda import linalg as cuda_linalg
    cuda_linalg.init()
    has_cuda = True
except ImportError:
    has_cuda = False

#has_cuda = False

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
        self.output_weights = maybe_cuda(np.zeros((
            self.n_output_units, self.n_internal_units + self.n_input_units)))

        scaled_feedback_weights = np.dot(self.feedback_weights, np.diag(self.feedback_scaling))
        scaled_feedback_weights = scaled_feedback_weights.reshape((len(self.feedback_weights), self.n_output_units))
        self.fixed_weights = maybe_cuda(np.hstack((self.internal_weights, self.input_weights, scaled_feedback_weights)))

        self.reset_state()

    def reset_state(self):
        self.total_state = np.zeros((self.n_input_units + self.n_internal_units +
                                     self.n_output_units, 1))
        self.internal_state = maybe_cuda(np.zeros((self.n_internal_units, 1)))
        self.internal_state = maybe_cuda(np.zeros((self.n_internal_units, 1)))

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
        self.output_weights = linear_regression_maybe_cuda(state_matrix, teacher_matrix)

        return state_matrix

    def test(self, input, n_forget_points=0, reset_points=None, actual_output=None):
        state_matrix = self._compute_state_matrix(input, n_forget_points=n_forget_points,
                                                  reset_points=reset_points, actual_output=actual_output)
        output = maybe_cuda_dot(maybe_cuda(state_matrix), self.output_weights.T)
        output = no_cuda(self.output_activation_function(output))
        output -= self.teacher_shift
        output /= self.teacher_scaling

        return output

    def train_multiple(self, inputs, outputs, n_forget_points=0):
        assert len(inputs) == len(outputs)
        state_matrix = None
        teacher_matrix = None
        for i, (input, output) in enumerate(itertools.izip(inputs, outputs)):
            print i
            assert input.shape[1] == self.n_input_units
            assert output.shape[1] == self.n_output_units

            local_state_matrix = self._compute_state_matrix(input, output, n_forget_points)
            local_teacher_matrix = self._compute_teacher_matrix(output, n_forget_points)

            if state_matrix is None:
                state_matrix = local_state_matrix
                teacher_matrix = local_teacher_matrix
            else:
                state_matrix = np.vstack((state_matrix, local_state_matrix))
                teacher_matrix = np.vstack((teacher_matrix, local_teacher_matrix))

            self.reset_state()

        self.output_weights = self._linear_regression_wiener_hopf(state_matrix, teacher_matrix)
        return state_matrix, teacher_matrix

    def _compute_state_matrix(self, input, output=None, n_forget_points=0,
                              reset_points=None, actual_output=None):
        state_matrix = np.zeros((len(input) - n_forget_points,
                                 self.n_input_units + self.n_internal_units))

        if self.callback:
            callback_state = np.zeros((self.callback_every, self.n_input_units + self.n_internal_units + self.n_output_units))

        if reset_points is not None:
            reset_points = set(reset_points)

        

        for i, input_point in enumerate(input):

            if reset_points is not None and i in reset_points:
                self.reset_state()

            scaled_input = (self.input_scaling * input_point + self.input_shift).reshape(
                len(input_point), 1)

            self.total_state[self.n_internal_units :
                             self.n_internal_units + self.n_input_units] = scaled_input
            if self.time_constants is None:
                self._update_internal_state()
            else:
                self._update_internal_state_leaky()

            self.total_state[:self.n_internal_units, :] = no_cuda(self.internal_state)
            self.total_state[self.n_internal_units:self.n_internal_units + self.n_input_units, :] = scaled_input

            if output is None:
                scaled_output = self.output_activation_function(
                    maybe_cuda_dot(self.output_weights, maybe_cuda(self.total_state[:-self.n_output_units, :])))
                scaled_output = no_cuda(scaled_output)
            else:
                scaled_output = self.teacher_scaling * output[i,:] + self.teacher_shift
                scaled_output = scaled_output.reshape((len(scaled_output), 1))

            self.total_state[-self.n_output_units:, :] = scaled_output

            if i >= n_forget_points:
                state_matrix[i - n_forget_points, :self.n_internal_units] = no_cuda(self.internal_state).T
                state_matrix[i - n_forget_points, self.n_internal_units:] = scaled_input.T

            if i % 100 == 0:
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

    def _update_internal_state_leaky(self):
        previous_internal_state = self.total_state[0:self.n_internal_units, :]
        scaled_feedback_weights = np.dot(self.feedback_weights, np.diag(self.feedback_scaling))
        scaled_feedback_weights = scaled_feedback_weights.reshape((len(self.feedback_weights), self.n_output_units))

        self.internal_state = ((1 - self.leakage * self.time_constants) * previous_internal_state +
                               self.time_constants * self.reservoir_activation_function(
                                   np.dot(np.hstack((self.internal_weights, self.input_weights, scaled_feedback_weights)),
                                          self.total_state)))

        self.internal_state += self.noise_level * (np.random.rand(self.n_internal_units, 1) - .5)

    def _update_internal_state(self):
        self.internal_state = self.reservoir_activation_function(maybe_cuda_dot(self.fixed_weights, maybe_cuda(self.total_state)))
        self.internal_state += self.noise_level * (maybe_cuda(np.random.rand(self.n_internal_units, 1)) - .5)

    def _compute_teacher_matrix(self, output, n_forget_points):
        teacher = maybe_cuda(self.teacher_scaling * output[n_forget_points:, :] + self.teacher_shift)
        return self.inverse_output_activation_function(teacher)

    def _generate_input_weights(self):
        return 2 * np.random.random((self.n_internal_units, self.n_input_units)) - 1

    def _generate_internal_weights(self):
        internal_weights = scipy.sparse.rand(
            self.n_internal_units, self.n_internal_units, self.connectivity)
        internal_weights = internal_weights.todense()
        return self._normalise_internal_weights(internal_weights)

    def _normalise_internal_weights(self, internal_weights):
        internal_weights[np.where(internal_weights != 0)] -= .5
        radius = np.max(np.abs(np.linalg.eigvals(internal_weights)))
        internal_weights /= radius
        internal_weights *= self.spectral_radius
        return internal_weights

    def _generate_feedback_weights(self):
        return 2 * np.random.random((self.n_internal_units, self.n_output_units)) - 1

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


class OnlineNeighbourESN(NeighbourESN):

    def train(self, input, output, n_forget_points=0, callback=None, callback_every=None, reset_points=None):
        assert len(input) == len(output)
        assert input.shape[1] == self.n_input_units
        assert output.shape[1] == self.n_output_units

        if callback:
            callback_state = np.zeros((callback_every, self.n_input_units + self.n_internal_units + self.n_output_units))

        if reset_points is not None:
            reset_points = set(reset_points)

        for i, input_point in enumerate(input):

            if reset_points is not None and i in reset_points:
                self.reset_state()

            scaled_input = (self.input_scaling * input_point + self.input_shift).reshape(
                len(input_point), 1)

            self.total_state[self.n_internal_units :
                             self.n_internal_units + self.n_input_units] = scaled_input
            if self.time_constants is None:
                self._update_internal_state()
            else:
                self._update_internal_state_leaky()

            scaled_output = self.teacher_scaling * output[i,:] + self.teacher_shift
            scaled_output = scaled_output.reshape((len(scaled_output), 1))

            self.total_state = np.vstack((self.internal_state, scaled_input, scaled_output))

            if i >= n_forget_points:
                # gradient descent

                estimated_output = self.output_activation_function(
                    np.dot(self.output_weights,
                           np.vstack((self.internal_state, scaled_input))))
                error = scaled_output - estimated_output
                    
                learning_rate = 0.012
                for o, w in enumerate(self.output_weights[0, :]):
                    x = self.total_state[o, 0]
                    dw = x * learning_rate * error
                    self.output_weights[0, o] += dw

            if callback:
                callback_state[i % callback_every,:] = np.vstack((scaled_input, self.internal_state, scaled_output)).T
                if (i + 1) % callback_every == 0:
                    #print i, len(input)
                    callback(callback_state)


def nrmse(estimated, correct):
    correct_variance = np.var(correct)
    if correct_variance == 0:
        correct_variance = 0.01 # hack

    return np.sqrt(mean_error(estimated, correct) / correct_variance)

def mean_error(estimated, correct):
    n_forget_points = len(correct) - len(estimated)
    correct = correct[n_forget_points:, :]
    return sum(np.power(estimated - correct, 2)) / len(estimated)

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

def maybe_cuda(x):
    if has_cuda:
        return gpuarray.to_gpu(x).astype('float32')
    else:
        return x

def no_cuda(x):
    if has_cuda:
        return x.get()
    else:
        return x

def maybe_cuda_dot(x, y):
    if has_cuda:
        return cuda_linalg.dot(x, y)
    else:
        return np.dot(x, y)

def cu_arctanh(x):
    return .5 * cumath.log((1 + x) / (1 - x))

def function_from_name(name, return_inverse=False):
    func = None
    inverse = None
    if name == 'identity':
        func = lambda x: x
        inverse = lambda x: x
    elif name == 'tanh':
        if has_cuda:
            func = cumath.tanh
            inverse = cu_arctanh
        else:
            func = np.tanh
            inverse = np.arctanh
    else:
        raise Exception('Unknown function: %s' % name)

    if return_inverse:
        return func, inverse
    else:
        return func

def linear_regression_maybe_cuda(state_matrix, teacher_matrix):
    if has_cuda:
        return cuda_linalg.transpose(maybe_cuda_dot(cuda_linalg.pinv(maybe_cuda(state_matrix)), teacher_matrix))
    else:
        run_length = np.shape(state_matrix)[0]
        cov_mat = state_matrix.T.dot(state_matrix / run_length)
        p_vec = state_matrix.T.dot(teacher_matrix / run_length)
        return (np.linalg.inv(cov_mat).dot(p_vec)).T
