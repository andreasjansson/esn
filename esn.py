import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

class EchoStateNetwork(object):

    def __init__(self,
                 n_input_units,
                 n_internal_units,
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
                 reservoir_activation_function=np.tanh,
                 output_activation_function='identity'):

        self.n_input_units = n_input_units
        self.n_internal_units = n_internal_units
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
        if time_constants is None:
            time_constants = [1] * n_internal_units
        self.time_constants = time_constants
        self.reservoir_activation_function = reservoir_activation_function

        if output_activation_function == 'identity':
            self.output_activation_function = lambda x: x
            self.inverse_output_activation_function = lambda x: x
        elif output_activation_function == 'tanh':
            self.output_activation_function = np.tanh
            self.inverse_output_activation_function = np.arctanh
        else:
            raise Exception('Unknown output activation function: %s' % output_activation_function)

        self.reset()

    def reset(self):
        self.input_weights = self._generate_input_weights()
        self.internal_weights = self._generate_internal_weights()
        self.feedback_weights = self._generate_feedback_weights()
        self.output_weights = np.zeros((
            self.n_output_units, self.n_internal_units + self.n_input_units))

    def train(self, input, output, n_forget_points=0):
        assert len(input) == len(output)
        assert input.shape[1] == self.n_input_units
        assert output.shape[1] == self.n_output_units

        state_matrix = self._compute_state_matrix(input, output, n_forget_points)
        teacher_matrix = self._compute_teacher_matrix(output, n_forget_points)
        self.output_weights = self._linear_regression_wiener_hopf(state_matrix, teacher_matrix)

        return state_matrix

    def test(self, input, n_forget_points=0):
        state_matrix = self._compute_state_matrix(input, n_forget_points=n_forget_points)
        output = state_matrix.dot(self.output_weights.T)
        output = self.output_activation_function(output)
        output -= self.teacher_shift
        output /= self.teacher_scaling

        return output

    def _compute_state_matrix(self, input, output=None, n_forget_points=0):
        state_matrix = np.zeros((len(input) - n_forget_points,
                                 self.n_input_units + self.n_internal_units))
        total_state = np.zeros((self.n_input_units + self.n_internal_units +
                                self.n_output_units, 1))
        internal_state = np.zeros((self.n_internal_units, 1))

        for i, input_point in enumerate(input):
            scaled_input = (self.input_scaling * input_point + self.input_shift).reshape(
                len(input_point), 1)

            total_state[self.n_internal_units :
                        self.n_internal_units + self.n_input_units] = scaled_input
            internal_state = self._update_internal_state(internal_state, total_state)

            if output is None:
                scaled_output = self.output_activation_function(
                    np.dot(self.output_weights,
                           np.vstack((internal_state, scaled_input))))
            else:
                scaled_output = self.teacher_scaling * output[i,:] + self.teacher_shift
                scaled_output = scaled_output.reshape((len(scaled_output), 1))

            total_state = np.vstack((internal_state, scaled_input, scaled_output))

            if i >= n_forget_points:
                state_matrix[i - n_forget_points,:] = np.vstack((internal_state, scaled_input)).T

        return state_matrix

    def _update_internal_state_leaky_UNTESTED(self, internal_state, total_state):
        total_weights = np.vstack((
            self.internal_weights, self.input_weights,
            self.feedback_weights * np.diag(self.feedback_scaling)))

        internal_state *= (1 - self.leakage * self.time_constants)
        internal_state += (self.time_constants * self.reservoir_activation_function(
            np.dot(total_weights * total_state)))

        internal_state += self.noise_level * np.random.random(
            (self.n_internal_units, 1)) - .5

    def _update_internal_state(self, internal_state, total_state):
        scaled_feedback_weights = np.dot(self.feedback_weights, np.diag(self.feedback_scaling))
        scaled_feedback_weights = scaled_feedback_weights.reshape((len(self.feedback_weights), self.n_output_units))
        internal_state = self.reservoir_activation_function(
            np.dot(np.hstack((self.internal_weights, self.input_weights, scaled_feedback_weights)),
                   total_state))
        internal_state += self.noise_level * (np.random.rand(self.n_internal_units, 1) - .5)
        return internal_state

    def _compute_teacher_matrix(self, output, n_forget_points):
        teacher = self.teacher_scaling * output[n_forget_points:, :] + self.teacher_shift
        return self.inverse_output_activation_function(teacher)

    def _linear_regression_pseudoinverse(self, state_matrix, teacher_matrix):
        return (np.linalg.pinv(state_matrix).dot(teacher_matrix)).T

    def _linear_regression_wiener_hopf(self, state_matrix, teacher_matrix):
        run_length = np.shape(state_matrix)[0]
        cov_mat = state_matrix.T.dot(state_matrix / run_length)
        p_vec = state_matrix.T.dot(teacher_matrix / run_length)
        return (np.linalg.inv(cov_mat).dot(p_vec)).T

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

class NeighbourESN(EchoStateNetwork):

    def __init__(self,
                 n_input_units,
                 n_internal_units,
                 n_output_units,
                 input_scaling,
                 input_shift,
                 teacher_scaling,
                 teacher_shift,
                 noise_level,
                 spectral_radius,
                 feedback_scaling,
                 feedback_every,
                 leakage=0,
                 time_constants=None,
                 reservoir_activation_function=np.tanh,
                 output_activation_function='identity'):

        self.feedback_every = feedback_every

        super(NeighbourESN, self).__init__(
            n_input_units=n_input_units,
            n_internal_units=n_internal_units,
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

    def height(self):
        height = int(np.sqrt(self.n_internal_units))
        assert self.n_internal_units % height == 0 # is rect
        return height

    def width(self):
        return self.n_internal_units / self.height()

    def point_to_index(self, x, y):
        return x % self.width() + y * self.width()

    def _generate_input_weights(self):
        weights = np.zeros((self.n_internal_units, self.n_input_units))
        left_column = 2 * np.random.random((self.height(), self.n_input_units)) - 1
        weights[np.arange(self.n_internal_units) % self.width() == 0, :] = left_column
        return weights

    def _generate_internal_weights(self):
        weights = np.zeros((self.n_internal_units, self.n_internal_units))

        for x in xrange(self.width() - 1):
            for y in xrange(self.height() - 1):
                neighbours = [(x + 1, y), (x, y + 1), (x + 1, y + 1)]
                for nx, ny in neighbours:
                    if np.random.rand() < .5:
                        start = (x, y)
                        end = (nx, ny)
                    else:
                        start = (nx, ny)
                        end = (x, y)
                    weights[self.point_to_index(*start),
                            self.point_to_index(*end)] = np.random.rand()

        return self._normalise_internal_weights(weights)

    def _get_feedback_indices(self):
        return ((np.arange(self.n_input_units) + 1) %
                (self.width() * self.feedback_every)) == 0

    def _generate_feedback_weights(self):
        weights = np.zeros((self.n_internal_units, self.n_output_units))
        feedback_indices = self._get_feedback_indices()
        feedback_weights = 2 * np.random.rand(
            len(feedback_indices), self.n_output_units) - 1
        weights[feedback_indices,:] = feedback_weights
        return weights

    def _generate_output_weights(self):
        weights = np.zeros((self.n_internal_units, self.n_output_units))
        output_indices = np.logical_not(self._get_feedback_indices())
        output_weights = 2 * np.random.rand(
            len(output_indices), self.n_output_units) - 1
        weights[output_indices,:] = output_weights
        return weights

def evaluate(esn, input, output, forget_points, iterations=10):
    best_output = None
    best_error = float('+inf')

    for i in xrange(iterations):
        if i > 0:
            esn.reset()

        esn.train(input, output, forget_points)
        estimated_output = esn.test(input)
        error = nrmse(estimated_output, output)
        if error < best_error:
            best_output = estimated_output
            best_error = error

    return best_output, best_error


def nrmse(estimated, correct):
    n_forget_points = len(correct) - len(estimated)
    correct = correct[n_forget_points:, :]
    correct_variance = np.var(correct)
    mean_error = sum(np.power(estimated - correct, 2)) / len(estimated)
    return np.sqrt(mean_error / correct_variance)
