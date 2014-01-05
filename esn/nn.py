import numpy as np

TINY = np.exp(-30)

class FeedForwardNetwork(object):

    def __init__(self, inputs, outputs, n_hidden=50, learning_rate=0.1,
                 initial_weight_stddev=0.01, momentum=0.9, batch_size=100):

        self.inputs = inputs
        self.outputs = outputs
        self.length, self.n_inputs = inputs.shape
        self.n_hidden = n_hidden
        self.n_outputs = outputs.shape[1]
        assert outputs.shape[0] == self.length
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.input_batches, self.n_batches = self.split_batches(inputs)

        self.input_to_hidden_weights = np.random.randn(
            self.n_inputs, self.n_hidden) * initial_weight_stddev
        self.hidden_to_output_weights = np.random.randn(
            self.n_hidden, self.n_outputs) * initial_weight_stddev
        self.hidden_bias = np.zeros(self.n_hidden)
        self.output_bias = np.zeros(self.n_outputs)

        self.input_to_hidden_delta = np.zeros((self.n_inputs, self.n_hidden))
        self.hidden_to_output_delta = np.zeros((self.n_hidden, self.n_outputs))
        self.hidden_bias_delta = np.zeros(self.n_hidden)
        self.output_bias_delta = np.zeros(self.n_outputs)
    
    def train(self, epochs=100):
        for epoch in range(epochs):
            total_cross_entropy = 0
            correct = 0
            for i, input_batch in enumerate(self.input_batches):
                output_batch = self.outputs[i * self.batch_size:(i + 1) * self.batch_size, :].T
                hidden_state, output_state = self.fprop(input_batch)

                cross_entropy = -np.sum(
                    np.sum(output_batch * np.log(output_state + TINY))) / self.batch_size
                total_cross_entropy += (cross_entropy - total_cross_entropy) / (i + 1)
                correct += np.sum(np.argmax(output_state, 0) == np.argmax(output_batch, 0))

                error_deriv = output_state - output_batch
                hidden_to_output_gradient = np.dot(hidden_state, error_deriv.T)
                output_bias_gradient = np.sum(error_deriv, 1)
                bp_deriv = np.dot(self.hidden_to_output_weights, error_deriv) * hidden_state * (1 - hidden_state)

                input_to_hidden_gradient = np.dot(input_batch.T, bp_deriv.T)
                hidden_bias_gradient = np.sum(bp_deriv, 1)

                self.input_to_hidden_delta *= self.momentum
                self.input_to_hidden_delta += input_to_hidden_gradient / self.batch_size
                self.input_to_hidden_weights -= self.learning_rate * self.input_to_hidden_delta

                self.hidden_to_output_delta *= self.momentum
                self.hidden_to_output_delta += hidden_to_output_gradient / self.batch_size
                self.hidden_to_output_weights -= self.learning_rate * self.hidden_to_output_delta

                self.hidden_bias_delta *= self.momentum
                self.hidden_bias_delta += hidden_bias_gradient / self.batch_size
                self.hidden_bias -= self.learning_rate * self.hidden_bias_delta

                self.output_bias_delta *= self.momentum
                self.output_bias_delta += output_bias_gradient / self.batch_size
                self.output_bias -= self.learning_rate * self.output_bias_delta

            print 'Epoch: %d, cross entropy: %.3f, correct: %.3f' % (
                epoch, total_cross_entropy, correct / float(self.length))

    def fprop(self, input_batch):
        # TODO: nicer way to add "horizonally"?
        inputs_to_hidden_units = (np.dot(
            self.input_to_hidden_weights.T, input_batch.T).T + self.hidden_bias).T
        hidden_state = 1 / (1 + np.exp(-inputs_to_hidden_units))
        inputs_to_softmax = (np.dot(
            self.hidden_to_output_weights.T, hidden_state).T + self.output_bias).T
        inputs_to_softmax -= np.max(inputs_to_softmax)
        output_state = np.exp(inputs_to_softmax)
        output_state /= np.sum(output_state, 0)

        return hidden_state, output_state

    def split_batches(self, inputs):
        length = inputs.shape[0]
        n_batches = length // self.batch_size
        input_batches = []
        for i in range(n_batches):
            input_batches.append(inputs[i * self.batch_size:(i + 1) * self.batch_size, :])
        return input_batches, n_batches
