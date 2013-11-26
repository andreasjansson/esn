import numpy as np
import cPickle
import cma

import esn
import test_data

if __name__ == '__main__':

    train_input, train_output, train_splits, test_input, test_output, test_splits, network = test_data.instrumentalness()

    optimiser = esn.GeneticOptimiser(network, train_input, train_output, 0)
    params = np.array(optimiser.initial_params())
    res = cma.fmin(optimiser.evaluate, params, 0.1, maxiter=20)

    filename = '/parent/best_esn_%s.pkl' % res[1]
    with open(filename, 'w') as f:
        cPickle.dump(network.serialize(), f)
