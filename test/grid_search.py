import numpy as np
import random
import itertools
import multiprocessing
import signal
import simplejson as json

import esn
import chord_recognition

input = output = split_points = None

def get_score(params):
    n_input_units=12
    n_output_units=len(chord_recognition.CHORD_MAP)
    network = esn.EchoStateNetwork(
        n_input_units=n_input_units,
        width=30,
        height=30,
        n_output_units=n_output_units,
        connectivity=params['connectivity'],
        input_scaling=[params['input_scaling']] * n_input_units,
        input_shift=[0] * n_input_units,
        noise_level=0.001,
        spectral_radius=params['spectral_radius'],
        feedback_scaling=[0] * n_output_units,
        leakage=params['leakage'],
        teacher_scaling=.99,
        output_activation_function='tanh'
    )

    network.train(input, output, reset_points=split_points, n_forget_points=0)
    network.noise_level = 0
    estimated_output = network.test(input, reset_points=split_points, n_forget_points=0)

    correct = np.sum(np.argmax(output, 1) == np.argmax(estimated_output, 1))

    score = correct / float(len(output))

#    print 'connectivity: %.2f' % params['connectivity']
#    print 'input_scaling: %.2f' % params['input_scaling']
#    print 'spectral_radius: %.2f' % params['spectral_radius']
#    print 'leakage: %.2f' % params['leakage']
    print 'correct: %.2f%%' % score

    return score

def main():
    global input, output, split_points

    # set OPENBLAS_NUM_THREADS before running!
    n_grid_threads = 32

    n_train = 50
    meta_data = chord_recognition.read_meta_data()
    ids = meta_data.keys()
    random.shuffle(ids)
    input, output, split_points = chord_recognition.read_data(ids[:n_train])

    grid = {
        'connectivity': [.01, .02, .05, .1, .3],
        'spectral_radius': [.8, .9, 1, 1.1, 1.2, 1.3],
        'leakage': [.02, .05, .1, .2],
        'input_scaling': [.5, .75, 1, 1.5],
    }

    items = sorted(grid.items())
    keys, values = zip(*items)
    grid_points = [dict(zip(keys, v)) for v in itertools.product(*values)]

    pool = multiprocessing.Pool(n_grid_threads)

    scores = pool.map(get_score, grid_points, chunksize=1)

    with open('scores.json', 'w') as f:
        json.dump(zip(grid_points, scores), f)

if __name__ == '__main__':
    main()
