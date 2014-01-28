import numpy as np
import random
import itertools
import multiprocessing
import signal
import simplejson as json
import os

import numpy as np
import random
random.seed(1)
np.random.seed(1)


import esn
import esn.wind
import chord_recognition

train_input = train_output = train_split_points = test_input = test_output = test_split_points = None

def get_score(params):

    width = height = 20
    n_input_units = 12
    n_output_units = len(chord_recognition.CHORD_MAP)
    network = esn.wind.Network(
        n_inputs=n_input_units,
        width=width,
        height=height,
        spectral_radius=params['spectral_radius'],
        leakage=params['leakage'],
        sharpness=params['sharpness'],
        damping=params['damping'],
        beta=params['beta'],
    )

    random.seed(params['seed'])
    np.random.seed(params['seed'])

    network.pretrain(train_input)
    #network.set_random_internal_weights(.05)
    network.fit(train_input, train_output, split_points=train_split_points)
    estimated_output = network.predict(test_input, split_points=test_split_points)

    correct = np.sum(np.argmax(test_output, 1) == np.argmax(estimated_output, 1))

    score = correct / float(len(test_output))

    import uuid
    filename = 'scores-%s-%s.json' % (score, str(uuid.uuid4()))

    with open(filename, 'w') as f:
        json.dump(params, f)
#    os.system('scp %s jansson.me.uk:~/scores/%s' % (filename, filename))

    print 'correct: %.2f%%' % score

    return score

def main():
    global test_input, test_output, test_split_points, train_input, train_output, train_split_points

    # set OPENBLAS_NUM_THREADS before running!
    n_grid_threads = 32

    (pretrain_input, pretrain_output, pretrain_split_points,
     train_input, train_output, train_split_points,
     test_input, test_output, test_split_points
    ) = chord_recognition.chord_data(1, 200, 50, return_notes=False)

    grid = {
        'spectral_radius': [.5, .9],
        'leakage': [.1, .2],
        'sharpness': [2],
        'damping': [.8],
        'beta': [0, 0.1, 1, 10],
        'seed': range(10),
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
