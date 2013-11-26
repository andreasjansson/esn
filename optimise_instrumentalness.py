import numpy as np
import cPickle
import cma

import esn
import test_data

def genetic_search():
    train_input, train_output, train_splits, test_input, test_output, test_splits, network = test_data.instrumentalness()

    optimiser = esn.GeneticOptimiser(network, train_input, train_output, 0)
    params = np.array(optimiser.initial_params())
    res = cma.fmin(optimiser.evaluate, params, 0.1, maxiter=20)

    filename = '/parent/best_esn_%s.pkl' % res[1]
    with open(filename, 'w') as f:
        cPickle.dump(network.serialize(), f)

def random_search():
    #train_input, train_output, train_splits, test_input, test_output, test_splits, network = test_data.instrumentalness()
    train_input, train_output, network = test_data.instrumentalness()

    optimiser = esn.GeneticOptimiser(network, train_input, train_output, 0)
    while True:
        error = optimiser.evaluate(optimiser.initial_params())
        filename = '/parent/esn_%s.pkl' % error
        with open(filename, 'w') as f:
            cPickle.dump(network.serialize(), f)
        network.reset()


if __name__ == '__main__':
    random_search()
