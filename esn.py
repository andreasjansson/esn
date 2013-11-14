import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import copy
import cma

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

        self.reset_state()

    def reset_state(self):
        self.total_state = np.zeros((self.n_input_units + self.n_internal_units +
                                     self.n_output_units, 1))
        self.internal_state = np.zeros((self.n_internal_units, 1))
        self.internal_state = np.zeros((self.n_internal_units, 1))

    def train(self, input, output, n_forget_points=0, callback=None, callback_every=None):
        assert len(input) == len(output)
        assert input.shape[1] == self.n_input_units
        assert output.shape[1] == self.n_output_units

        state_matrix = self._compute_state_matrix(input, output, n_forget_points,
                                                  callback=callback, callback_every=callback_every)
        teacher_matrix = self._compute_teacher_matrix(output, n_forget_points)
        self.output_weights = self._linear_regression_wiener_hopf(state_matrix, teacher_matrix)

        return state_matrix

    def test(self, input, n_forget_points=0, callback=None, callback_every=None):
        state_matrix = self._compute_state_matrix(input, n_forget_points=n_forget_points,
                                                  callback=callback, callback_every=callback_every)
        output = state_matrix.dot(self.output_weights.T)
        output = self.output_activation_function(output)
        output -= self.teacher_shift
        output /= self.teacher_scaling

        return output

    def _compute_state_matrix(self, input, output=None, n_forget_points=0,
                              callback=None, callback_every=None):
        state_matrix = np.zeros((len(input) - n_forget_points,
                                 self.n_input_units + self.n_internal_units))

        if callback:
            callback_state = np.zeros((callback_every, self.n_input_units + self.n_internal_units + self.n_output_units))

        for i, input_point in enumerate(input):
            scaled_input = (self.input_scaling * input_point + self.input_shift).reshape(
                len(input_point), 1)

            self.total_state[self.n_internal_units :
                             self.n_internal_units + self.n_input_units] = scaled_input
            self._update_internal_state()

            if output is None:
                scaled_output = self.output_activation_function(
                    np.dot(self.output_weights,
                           np.vstack((self.internal_state, scaled_input))))
            else:
                scaled_output = self.teacher_scaling * output[i,:] + self.teacher_shift
                scaled_output = scaled_output.reshape((len(scaled_output), 1))

            self.total_state = np.vstack((self.internal_state, scaled_input, scaled_output))

            if i >= n_forget_points:
                state_matrix[i - n_forget_points,:] = np.vstack((self.internal_state, scaled_input)).T

            if callback:
                callback_state[i % callback_every,:] = np.vstack((scaled_input, self.internal_state, scaled_output)).T
                if (i + 1) % callback_every == 0:
                    callback(callback_state)

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

    def _update_internal_state(self):
        scaled_feedback_weights = np.dot(self.feedback_weights, np.diag(self.feedback_scaling))
        scaled_feedback_weights = scaled_feedback_weights.reshape((len(self.feedback_weights), self.n_output_units))
        self.internal_state = self.reservoir_activation_function(
            np.dot(np.hstack((self.internal_weights, self.input_weights, scaled_feedback_weights)),
                   self.total_state))
        self.internal_state += self.noise_level * (np.random.rand(self.n_internal_units, 1) - .5)

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

        self.width = width
        self.height = height
        n_internal_units = width * height

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

    def get_input_weight(self, i, x2, y2):
        return self.input_weights[self.point_to_index(x2, y2), i]

    def get_internal_weight(self, x1, y1, x2, y2):
        return self.internal_weights[self.point_to_index(x1, y1), self.point_to_index(x2, y2)]

    def get_feedback_weight(self, i, x2, y2):
        return self.feedback_weights[self.point_to_index(x2, y2), i]

    def get_output_weight(self, i, x2, y2):
        return self.output_weights[i, self.point_to_index(x2, y2)]

    def point_to_index(self, x, y):
        return x % self.width + y * self.width

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
    n_forget_points = len(correct) - len(estimated)
    correct = correct[n_forget_points:, :]
    correct_variance = np.var(correct)
    mean_error = sum(np.power(estimated - correct, 2)) / len(estimated)
    return np.sqrt(mean_error / correct_variance)

def optimise(esn, input, output, forget_points, iterations=10, test=nrmse):
    best_output = None
    best_error = float('+inf')

    for i in xrange(iterations):
        if i > 0:
            esn.reset()

        esn.train(input, output, forget_points)
        estimated_output = esn.test(input)
        error = np.sum(nrmse(estimated_output, output))
        if error < best_error:
            print i, error
            best_output = estimated_output
            best_error = error
            best_esn = copy.deepcopy(esn)

    return best_esn, best_output, best_error
    

class MyFloat(float):
    output = None
    estimated_output = None

class GeneticOptimiser(object):

    def __init__(self, esn, input, output, forget_points, test=nrmse):
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
#        error.output = self.output
#        error.estimated_output = estimated_output

        return error

    
def mutate(params, mutate_ratio=0.2):
    ndx = np.random.choice(len(params), int(len(params) * mutate_ratio))
    values = params[ndx]
    np.random.shuffle(values)
    values += np.random.rand(len(values)) * .1
    params[ndx] = values
    return params.copy()

def mate(params1, params2):
    split = np.random.randint(len(params1))
    return np.concatenate((params1[:split], params2[split:])), np.concatenate((params1[split:], params2[:split]))

def fmin(func, params, pool_size=20, generation=0):
    print 'generation: %d' % generation
    pool = [None] * pool_size
    errors = [0] * pool_size
    for i in range(pool_size):
        pool[i] = mutate(params)
        errors[i] = func(pool[i])

    best_ndx = np.argsort(errors)[:4]
    np.random.shuffle(best_ndx)
    for i in range(0, len(best_ndx), 2):
        i1, i2 = best_ndx[i], best_ndx[i + 1]
        print errors[i1], errors[i2]
        offspring = mate(pool[i1], pool[i2])

    pool = None
    errors = None
    for child in offspring:
        fmin(func, child, pool_size, generation + 1)

    import ipdb; ipdb.set_trace()
