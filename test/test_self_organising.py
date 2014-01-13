import numpy as np
import scipy.sparse
import random

from esn import selforganising
import chord_recognition

import matplotlib.pyplot as plt
plt.ion()

def test_data(sequence_length=2000, min_freq=20, max_freq=100, sr=1000):

    n_outputs = 5

    freqs = np.random.rand(n_outputs) * (max_freq - min_freq) + min_freq
    current_freq = np.random.randint(n_outputs)

    phase = 0
    outputs = np.zeros((sequence_length, n_outputs))
    audio = np.zeros((sequence_length, 1))

    for i in xrange(1, sequence_length):

        if np.random.rand() < 0.003:
            current_freq = np.random.randint(n_outputs)

        outputs[i, current_freq] = 1

        freq = freqs[current_freq]
        phase += 2 * np.pi / sr
        if phase > 2 * np.pi:
            phase -= 2 * np.pi
        audio[i, 0] = np.sin(phase * freq)

    audio = (audio + 1) / 2

    return audio, outputs, audio, outputs, audio, outputs

def chord_data():
    n_pretrain = 100
    n_train = 100
    n_test = 30

    meta_data = chord_recognition.read_meta_data()
    ids = meta_data.keys()
    random.shuffle(ids)
    pretrain_input, pretrain_output, pretrain_split_points = chord_recognition.read_data(ids[:n_pretrain])
    train_input, train_output, train_split_points = chord_recognition.read_data(ids[n_pretrain:n_pretrain + n_train])
    test_input, test_output, test_split_points = chord_recognition.read_data(ids[n_pretrain + n_train:n_pretrain + n_train + n_test])

    return pretrain_input, pretrain_output, train_input, train_output, test_input, test_output

pretrain_inputs, pretrain_outputs, train_inputs, train_outputs, test_inputs, test_outputs = chord_data()
#pretrain_inputs, pretrain_outputs, train_inputs, train_outputs, test_inputs, test_outputs = test_data()
test_inputs, test_outputs = train_inputs, train_outputs

n_inputs = train_inputs.shape[1]
n_outputs = train_outputs.shape[1]
width = 50
height = 3
n_internal = width * height

network = selforganising.DeepSelfOrganisingReservoir(
    n_inputs=n_inputs,
    n_outputs=n_outputs,
#    sizes=[(width, height), (width, height), (width, height)],
    sizes=[(width, height)],
    input_scaling=100,
    internal_scaling=50,
    leakage=.2,
    learning_rate_start=.05,
    neighbourhood_width_start=2,
)


def _generate_internal_weights(connectivity=.1):
    internal_weights = scipy.sparse.rand(
        n_internal, n_internal, connectivity).tocsr()
    return _normalise_internal_weights(internal_weights)

def _normalise_internal_weights(internal_weights, spectral_radius=.8):
    internal_weights.data -= .5
    attempts = 5
    for i in range(attempts):
        try:
            eigvals = scipy.sparse.linalg.eigs(
                internal_weights, k=1, which='LM',
                return_eigenvectors=False, tol=.02, maxiter=5000)
            break
        except (scipy.sparse.linalg.ArpackNoConvergence,
                scipy.sparse.linalg.ArpackError):
            continue
    else:
        print 'scipy.sparse.linalg failed to converge, falling back to numpy.linalg'
        eigvals = np.linalg.eigvals(internal_weights.todense())
    radius = np.abs(np.max(eigvals))
    internal_weights /= radius
    internal_weights *= spectral_radius
    return np.array(internal_weights.todense())

first_layer = network.layers[0]

for layer in network.layers:
    layer.internal_weights = _generate_internal_weights()
    layer.input_weights = 2 * np.random.rand(layer.n_inputs, layer.n_internal) - 1

first_layer.fit(train_inputs, train_outputs)
predicted2 = first_layer.predict(test_inputs)

network.pretrain(pretrain_inputs)
network.fit(train_inputs, train_outputs)
predicted = network.predict(test_inputs)

import operator
for p, a in zip(predicted, test_outputs)[:1000]:
    p_probs = chord_recognition.get_chord_probs(p)
    a_probs = chord_recognition.get_chord_probs(a)
    print '%s %s %s' % (a_probs[0][0], '==' if a_probs[0][0] == p_probs[0][0] else '!=', [(x, round(y, 2)) for x, y in sorted(p_probs[:3], key=operator.itemgetter(1), reverse=True)])

import ipdb; ipdb.set_trace()

in3top = sum([(chord_recognition.get_chord_probs(test_outputs[i])[0][0] in dict(chord_recognition.get_chord_probs(predicted[i])[:3])) for i in range(len(predicted))])
correct = sum([(chord_recognition.get_chord_probs(test_outputs[i])[0][0] in dict(chord_recognition.get_chord_probs(predicted[i])[:1])) for i in range(len(predicted))])

in3top2 = sum([(chord_recognition.get_chord_probs(test_outputs[i])[0][0] in dict(chord_recognition.get_chord_probs(predicted2[i])[:3])) for i in range(len(predicted))])
correct2 = sum([(chord_recognition.get_chord_probs(test_outputs[i])[0][0] in dict(chord_recognition.get_chord_probs(predicted2[i])[:1])) for i in range(len(predicted))])

plt.imshow(network.layers[-1].internal_weights, interpolation='none')

