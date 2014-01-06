import numpy as np
import random

random.seed(1)
np.random.seed(1)

import chord_recognition
import esn

n_train = 20
n_test = 5
meta_data = chord_recognition.read_meta_data()
ids = meta_data.keys()
#random.shuffle(ids)
train_ids = ids[:n_train]
test_ids = ids[n_train:n_train + n_test]
train_inputs, train_outputs, split_train_points = chord_recognition.read_data(ids[:n_train])
test_inputs, test_outputs, test_split_points = chord_recognition.read_data(ids[n_train:n_train + n_test])

n_input_units = train_inputs.shape[1]
n_output_units = len(chord_recognition.CHORD_MAP)

input_scaling = [0.75] * 12
input_shift = [-0.25] * 12

width = height = 20

network = esn.EchoStateNetwork(
    n_input_units=n_input_units,
    width=width,
    height=height,
    connectivity=0.05,
    n_output_units=n_output_units,
    input_scaling=input_scaling,
    input_shift=input_shift,
    #noise_level=0.001,
    noise_level=0,
    spectral_radius=1.1,
    feedback_scaling=[0] * n_output_units,
    leakage=np.array([.2] * (width * height * 2/4) + [.5] * (width * height * 2/4)),
    teacher_scaling=.99,
    output_activation_function='tanh'
)

state_matrix = network.train(train_inputs, train_outputs)
state_matrix2 = network.compute_state_matrix(test_inputs, test_outputs)

nn = esn.FeedForwardNetwork(state_matrix.shape[1], train_outputs.shape[1], n_hidden=40)

nn.train(state_matrix, train_outputs, 50, state_matrix2, test_outputs)

test_output = network.test(test_inputs)

output_state = nn.test(state_matrix2, test_outputs)

correct_nn = np.sum(np.argmax(output_state, 0) == np.argmax(test_outputs, 1))
correct_lr = np.sum(np.argmax(test_output, 1) == np.argmax(test_outputs, 1))

import ipdb; ipdb.set_trace()

import ipdb; ipdb.set_trace()
