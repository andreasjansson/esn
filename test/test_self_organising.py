import numpy as np
import scipy.sparse
import random

random.seed(1)
np.random.seed(1)

from esn import selforganising, som2
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
    n_pretrain = 10
    n_train = 10
    n_test = 10

    meta_data = chord_recognition.read_meta_data()
    ids = meta_data.keys()
    random.shuffle(ids)
    pretrain_inputs, pretrain_outputs, pretrain_split_points = chord_recognition.read_data(ids[:n_pretrain])
    train_inputs, train_outputs, train_split_points = chord_recognition.read_data(ids[n_pretrain:n_pretrain + n_train])
    test_inputs, test_outputs, test_split_points = chord_recognition.read_data(ids[n_pretrain + n_train:n_pretrain + n_train + n_test])

    return pretrain_inputs, pretrain_outputs, pretrain_split_points, train_inputs, train_outputs, train_split_points, test_inputs, test_outputs, test_split_points

pretrain_inputs, pretrain_outputs, pretrain_split_points, train_inputs, train_outputs, train_split_points, test_inputs, test_outputs, test_split_points = chord_data()
#test_inputs, test_outputs = train_inputs, train_outputs

n_inputs = train_inputs.shape[1]
n_outputs = train_outputs.shape[1]


# ideas: connect all adjacent nodes and treat as echo state network
# 
#        2-dimensional grid, where column is time, self-organise
#        x[t-n], [...], x[t] over columns (with wrap-around). potentially
#        m times slower, where m is the number of columns. m may be > n.
#        alternatively, m = n and no wrap around (much faster). this is
#        effectively the same as having [x[t-n] ; [...] ; x[t]] as inputs.
#        we could have [(x[t-n] + [...] + x[t-1])/(n - 1) ; x[t]] as inputs
#        for improved speed (and perhaps generalisation).
#
#        automatically connect current and previous best matching unit
#        during pretraining (even if itself).

network = som2.SOM2(n_inputs, width=20, height=20, neighbourhood_width_start=3, neighbourhood_width_end=1, learning_rate_start=0.04, learning_rate_end=0.01)
network.pretrain(pretrain_inputs, pretrain_split_points)
network.fit(train_inputs, train_outputs, train_split_points)
predicted = network.predict(test_inputs, test_split_points)

import ipdb; ipdb.set_trace()

width = 100
height = 1
n_internal = width * height

network = selforganising.DeepSelfOrganisingReservoir(
    n_inputs=n_inputs,
    n_outputs=n_outputs,
#    sizes=[(width, height), (width, height), (width, height)],
    sizes=[(width, height)],
    input_scaling=100,
    internal_scaling=50,
    leakage=.5,
    learning_rate_start=.05,
    neighbourhood_width_start=10,
)


first_layer = network.layers[0]

for layer in network.layers:
    layer.input_weights = 2 * np.random.rand(layer.n_inputs, layer.n_internal) - 1

first_layer.fit(train_inputs, train_outputs, train_split_points)
predicted2 = first_layer.predict(test_inputs, test_split_points)

network.pretrain(pretrain_inputs, pretrain_split_points)
network.fit(train_inputs, train_outputs, train_split_points)
predicted = network.predict(test_inputs, test_split_points)

import operator
for p, a in zip(predicted, test_outputs)[:1000]:
    p_probs = chord_recognition.get_chord_probs(p)
    a_probs = chord_recognition.get_chord_probs(a)
    print '%s %s %s' % (a_probs[0][0], '==' if a_probs[0][0] == p_probs[0][0] else '!=', [(x, round(y, 2)) for x, y in sorted(p_probs[:3], key=operator.itemgetter(1), reverse=True)])

plt.imshow(network.layers[-1].internal_weights, interpolation='none')

in3top = sum([(chord_recognition.get_chord_probs(test_outputs[i])[0][0] in dict(chord_recognition.get_chord_probs(predicted[i])[:3])) for i in range(len(predicted))])
correct = sum([(chord_recognition.get_chord_probs(test_outputs[i])[0][0] in dict(chord_recognition.get_chord_probs(predicted[i])[:1])) for i in range(len(predicted))])

in3top2 = sum([(chord_recognition.get_chord_probs(test_outputs[i])[0][0] in dict(chord_recognition.get_chord_probs(predicted2[i])[:3])) for i in range(len(predicted))])
correct2 = sum([(chord_recognition.get_chord_probs(test_outputs[i])[0][0] in dict(chord_recognition.get_chord_probs(predicted2[i])[:1])) for i in range(len(predicted))])

import ipdb; ipdb.set_trace()
