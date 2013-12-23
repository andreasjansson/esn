# TODO: what happens if you take the estimated output and feed it into
# a second layer network, trying to predict the same output?
# what happens if that network also has timbre information as input?
# or some lower-dimensional features, like energy or loudness?

# TODO:
# * bias input (with different scaling)
# * chord overlap ratio / weighted chord overlap ratio (mirex evaluation)
# * remove smaller principal components
# * experiment with band-pass filters (especially when training on audio, similar to auditory models)
# * train on audio
# * compare the performance of single to layered reservoirs
# * sub-populations with different leaking rates
# * use ridge regression, logarithmic grid search on test data to find \beta
# * no noise necessary when using ridge regression
# * make sure resetting between sequences actually works
# * perhaps run sequences in parallel, if sparse matrices are faster
# * try logistic regression instead of a second reservoir to improve the estimated output (obviously lacking history)
# * use hmm on final outputs
# * include echo nest tibre vector in input
# * bandpass filters
# * bandpass filtered different unconnected esn
# * stack networks
# * treat stacked networks as a mlp

import esn
import numpy as np
import simplejson as json
import csv
import matplotlib.pyplot as plt
import os
import time
import random

DATA_DIR = os.path.expanduser('~/data/billboard')

ENHARMONIC_EQUIVALENTS = {
    'Db': 'C#',
    'Eb': 'D#',
    'Fb': 'E',
    'E#': 'F',
    'Gb': 'F#',
    'Ab': 'G#',
    'Bb': 'A#',
    'Cb': 'B',
    'B#': 'C',
}
CHORD_MAP = {
    'C:maj': 0,
    'C:min': 1,
    'C#:maj': 2,
    'C#:min': 3,
    'D:maj': 4,
    'D:min': 5,
    'D#:maj': 6,
    'D#:min': 7,
    'E:maj': 8,
    'E:min': 9,
    'F:maj': 10,
    'F:min': 11,
    'F#:maj': 12,
    'F#:min': 13,
    'G:maj': 14,
    'G:min': 15,
    'G#:maj': 16,
    'G#:min': 17,
    'A:maj': 18,
    'A:min': 19,
    'A#:maj': 20,
    'A#:min': 21,
    'B:maj': 22,
    'B:min': 23,
    'N': 24,
    'X': 25,
}

def run(train_input, train_output, train_split_points, test_input, test_output, test_split_points):

    n_input_units = train_input.shape[1]
    n_output_units = len(CHORD_MAP)

    if n_input_units == 24:
        input_scaling = [0.75] * 12 + [.01] * 12
        input_shift = [-0.25] * 12 + [0] * 12
    elif n_input_units == 12:
        input_scaling = [0.75] * 12
        input_shift = [-0.25] * 12
    elif n_input_units == 13:
        input_scaling = [0.75] * 12 + [.05]
        input_shift = [-0.25] * 12 + [.2]

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

    if hasattr(esn, 'Visualiser'):
        del esn.Visualiser
    if hasattr(esn, 'Visualiser'):
        visualiser = esn.Visualiser(network, 1000, input_yscale=.5, internal_yscale=.5, output_yscale=.5)

    n_forget_points = 0

    network.train(train_input, train_output, reset_points=train_split_points, n_forget_points=n_forget_points)

    network.noise_level = 0

    if hasattr(esn, 'Visualiser'):
        visualiser.set_weights()

    estimated_train_output = network.test(train_input, n_forget_points=n_forget_points, reset_points=train_split_points, actual_output=train_output)
    train_error = esn.nrmse(estimated_train_output, train_output[n_forget_points:])
    train_correct = np.sum(np.argmax(train_output[n_forget_points:], 1) == np.argmax(estimated_train_output, 1))
    train_accuracy = train_correct / float(len(estimated_train_output))

    estimated_test_output = network.test(test_input, n_forget_points=n_forget_points, reset_points=test_split_points, actual_output=test_output)
    test_error = esn.nrmse(estimated_test_output, test_output[n_forget_points:])
    test_correct = np.sum(np.argmax(test_output[n_forget_points:], 1) == np.argmax(estimated_test_output, 1))
    test_accuracy = test_correct / float(len(estimated_test_output))

    return estimated_train_output, estimated_test_output, train_accuracy, test_accuracy

def main():
    n_train = 100
    n_test = 50
    meta_data = read_meta_data()
    ids = meta_data.keys()
    random.shuffle(ids)
    train_ids = ids[:n_train]
    test_ids = ids[n_train:n_train + n_test]
    train_input, train_output, train_split_points = read_data(ids[:n_train])
    test_input, test_output, test_split_points = read_data(ids[n_train:n_train + n_test])

    n_forget_points = 0

    estimated_train_output, estimated_test_output, train_accuracy, test_accuracy = run(train_input, train_output, train_split_points, test_input, test_output, test_split_points)

    import ipdb; ipdb.set_trace()

    estimated_train_output2, estimated_test_output2, train_accuracy2, test_accuracy2 = run(estimated_train_output, train_output, train_split_points, estimated_test_output, test_output, test_split_points)

    print '######## train accuracy: %f%%' % train_accuracy
    print '######## test accuracy: %f%%' % test_accuracy

    from esn import postprocess

    import ipdb; ipdb.set_trace()

    CHORD_INDEX = {v: k for k, v in CHORD_MAP.items()}
    [(CHORD_INDEX[i], CHORD_INDEX[j], i == j) for i, j in zip(np.argmax(test_output[4000:5000 ], 1), np.argmax(estimated_test_output[4000:5000 ], 1))]

    plt.imshow(estimated_test_output[:1000, :], aspect='auto', interpolation='none')

    plt.plot(np.argmax(test_output[n_forget_points: ], 1))
    plt.plot(np.argmax(estimated_test_output, 1))
    plt.show()

    import ipdb; ipdb.set_trace()


def read_meta_data():
    meta_data = {}
    with open('%s/billboard-2.0-index.csv' % DATA_DIR) as f:
        for row in csv.reader(f):
            id, chart_date, target_rank, actual_rank, title, artist, peak_rank, weeks_on_chart = row
            if id == 'id':
                continue
            id = int(id)
            if artist and title:
                meta_data[id] = (artist, title)
    return meta_data

def read_data(ids):
    all_chromas = []
    all_chords = []
    split_points = []
    for id in ids:
        timed_chromas = parse_chroma_file('%s/McGill-Billboard/%04d/echonest.json' % (DATA_DIR, id))
        timed_chords = parse_chords_file('%s/McGill-Billboard/%04d/majmin.lab' % (DATA_DIR, id))
        chromas, chords = combine_chromas_and_chords(timed_chromas, timed_chords)
        all_chromas += chromas
        all_chords += chords
        split_points.append(len(all_chromas))

    return np.array(all_chromas), np.array(all_chords), split_points

def parse_chroma_file(filename, with_timbre=False, with_loudness=False):
    timed_chromas = []
    max_loudness = float('-inf')
    min_loudness = float('inf')
    with open(filename) as f:
        raw = json.load(f)
        segments = raw['segments']
        for segment in segments:
            start = segment['start']
            chroma = segment['pitches']

            if with_timbre:
                timbre = segment['timbre']
                chroma += timbre

            if with_loudness or 1:
                loudness = segment['loudness_max']
                loudness = (loudness + 60) / 60.0
                chroma = [c * loudness for c in chroma]
#                chroma += [loudness]

            timed_chromas.append((float(start), chroma))
    return timed_chromas

def get_zweiklangs(chroma):
    peaks = np.argsort(chroma)[-2:]
    chroma = np.zeros(len(chroma))
    chroma[peaks] = 1
    return chroma

def parse_chords_file(filename):
    timed_chords = []
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                start, end, chord = line.split('\t')
                chord_vector = normalise_chord(chord)
                timed_chords.append((float(start), chord_vector))
    return timed_chords

def normalise_chord(chord):
    if ':' in chord:
        root, mode = chord.split(':')
        if root in ENHARMONIC_EQUIVALENTS:
            root = ENHARMONIC_EQUIVALENTS[root]
            chord = '%s:%s' % (root, mode)
    vector = np.zeros(len(CHORD_MAP))
    vector[CHORD_MAP[chord]] = 1
    return vector

def combine_chromas_and_chords(timed_chromas, timed_chords):
    chromas = []
    chords = []
    chord_ndx = 0
    for i, (chroma_time, chroma) in enumerate(timed_chromas):
        chromas.append(chroma)

        if i == len(timed_chromas) - 1:
            chroma_duration = float('+inf')
        else:
            chroma_duration = timed_chromas[i + 1][0] - chroma_time

        chord_time, chord = timed_chords[chord_ndx]
        if chord_ndx == len(timed_chords) - 1:
            chord_duration = float('+inf')
        else:
            chord_duration = timed_chords[chord_ndx + 1][0] - chord_time

        if chord_time + chord_duration < chroma_time + chroma_duration / 2:
            chord_ndx += 1
            _, chord = timed_chords[chord_ndx]

        chords.append(chord)

    return chromas, chords

if __name__ == '__main__':
    main()
