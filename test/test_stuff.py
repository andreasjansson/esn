import numpy as np
import chord_recognition

import matplotlib.pyplot as plt
plt.ion()

(pretrain_inputs, pretrain_outputs, pretrain_split_points,
 train_inputs, train_outputs, train_split_points,
 test_inputs, test_outputs, test_split_points
) = chord_recognition.chord_data(1, 3, 50, return_notes=False)

import esn.wind
network = esn.wind.Network(12, 20, 20)
network.pretrain(pretrain_inputs)

import ipdb; ipdb.set_trace()

templates = np.zeros((len(chord_recognition.CHORD_NOTES), 12))
for name, i in chord_recognition.CHORD_MAP.items():
    for note in chord_recognition.CHORD_NOTES[name]:
        templates[i, note] = 1

matched = templates.dot(train_inputs.T).T

inputs = train_inputs
n_inputs = inputs.shape[1]

import scipy.sparse

cycle_lengths = (np.sqrt(2) ** (np.arange(8) + 8)).astype(int)
cycle_lengths = [64, 128]
n_internal = sum(cycle_lengths)

internal_weights = scipy.sparse.csr_matrix((n_internal, n_internal))
start = 0
weight = .99

for length in cycle_lengths:
    for i in np.arange(length):
        internal_weights[i + start, ((i + 1) % length) + start] = weight
        if np.random.rand() > .5:
            internal_weights[i + start, ((i + 1) % length) + start] = -weight
    start += length

#for _ in xrange(20):
#    if np.random.rand() > .5:
#        internal_weights[np.random.randint(n_internal), np.random.randint(n_internal)] = weight
#    else:
#        internal_weights[np.random.randint(n_internal), np.random.randint(n_internal)] = -weight

#internal_weights = scipy.sparse.rand(n_internal, n_internal, density=0.05)
#internal_weights.data -= .5
#internal_weights.data *= .8

input_weights = np.random.rand(n_inputs, n_internal) - .5

split_points = pretrain_split_points

activations = np.zeros((n_internal, 1))
history = np.zeros((len(inputs), n_internal))

fixed_weights = scipy.sparse.vstack((input_weights, internal_weights))

leakage = .5

for i, x in enumerate(inputs):
    x = x[np.newaxis].T

    if i % 1000 == 0:
        print '%d/%d' % (i, len(inputs))
        plt.clf()
        plt.plot(activations)
        plt.draw()


    if split_points is not None and i in split_points:
        activations = np.zeros((n_internal, 1))

    total_state = np.vstack((x, activations))
    activations = leakage * fixed_weights.T.dot(total_state) + (1 - leakage) * activations

    history[i, :] = activations.T

def linear_regression(history, outputs, beta=0):
    return outputs.T.dot(history).dot(np.linalg.inv(history.T.dot(history) + beta)).T

#plt.imshow(history, aspect='auto', interpolation='none')

output_weights = linear_regression(history, train_outputs)

actual = np.argmax(train_outputs, 1)
template_predicted = np.argmax(matched, 1)
train_predicted = np.argmax(history.dot(output_weights), 1)

import ipdb; ipdb.set_trace()

