# TODO: what happens if you take the estimated output and feed it into
# a second layer network, trying to predict the same output?
# what happens if that network also has timbre information as input?
# or some lower-dimensional features, like energy or loudness?

# TODO:
# * bias input (with different scaling)
# * remove smaller principal components
# * experiment with band-pass filters (especially when training on audio, similar to auditory models)
# * train on audio
# * compare the performance of single to layered reservoirs
# * sub-populations with different leaking rates
# * record performance of parameters so as to not go in circles
# * grid search for parameters (what is grid search ???)
# * use ridge regression, logarithmic grid search (?) on test data to find \beta
# * no noise necessary when using ridge regression
# * make sure resetting between sequences actually works
# * perhaps run sequences in parallel, if sparse matrices are faster
# * try logistic regression instead of a second reservoir to improve the estimated output (obviously lacking history)
# * use hmm on final outputs

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

def main():
    n_input_units = 12
    n_output_units = len(CHORD_MAP)

    import ipdb; ipdb.set_trace()

    import statprof
    statprof.start()
    network = esn.EchoStateNetwork(
        n_input_units=n_input_units,
        width=50,
        height=50,
        connectivity=0.01,
        n_output_units=n_output_units,
        input_scaling=[2] * n_input_units,
        input_shift=[-.5] * n_input_units,
        noise_level=0.001,
        spectral_radius=.1,
        feedback_scaling=[0] * n_output_units,
        leakage=.1,
        teacher_scaling=.99,
        output_activation_function='tanh'
    )
    statprof.stop()
    statprof.display()
    import ipdb; ipdb.set_trace()

    t0 = time.time()

    n_train = 50
    n_test = 20
    meta_data = read_meta_data()
    ids = meta_data.keys()
    random.shuffle(ids)
    train_ids = ids[:n_train]
    test_ids = ids[n_train:n_train + n_test]
    input, output, split_points = read_data(ids[:n_train])

    if hasattr(esn, 'Visualiser'):
        del esn.Visualiser
    if hasattr(esn, 'Visualiser'):
        visualiser = esn.Visualiser(network, 1000, input_yscale=.5, internal_yscale=.05, output_yscale=.5)

    n_forget_points = 0

    network.train(input, output, reset_points=split_points, n_forget_points=n_forget_points)

    network.noise_level = 0

    if hasattr(esn, 'Visualiser'):
        visualiser.set_weights()

    estimated_output = network.test(
        input, n_forget_points=n_forget_points, reset_points=split_points, actual_output=output)
        

    error = esn.nrmse(estimated_output, output[n_forget_points:])

#    plt.plot(np.argmax(output[n_forget_points: ], 1))
#    plt.plot(np.argmax(estimated_output, 1))
#    plt.show()

    total_time = time.time() - t0

    correct = np.sum(np.argmax(output, 1) == np.argmax(estimated_output, 1))

    print '######## total time: %f' % total_time
    print '######## correct: %f (%f%%)' % (correct, correct / float(len(output)))
    
#    plt.plot(np.argmax(output, 1))
#    plt.plot(np.argmax(estimated_output, 1))
#    plt.plot(np.argmax(lr_output, 1))
#    plt.show()

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

def parse_chroma_file(filename):
    timed_chromas = []
    with open(filename) as f:
        raw = json.load(f)
        segments = raw['segments']
        for segment in segments:
            start = segment['start']
            chroma = segment['pitches']
#            chroma = get_zweiklangs(chroma)
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
