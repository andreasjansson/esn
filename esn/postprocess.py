import numpy as np

def get_transition_model(sequences, height=24):
    # TODO: take into account previous chord as well as an unordered
    # set of the previous chords over the past n time steps
    model = np.ones((height, height)) # laplace smoothing
    for seq in sequences:
        for x, y in zip(seq[:-1], seq[1:]):
            model[x, y] += 1
    return model

def find_sequence(output, model):
    length, height = output.shape

    costs = np.zeros((length, height))
    prev = np.zeros((length, height))
    costs[0, :] = output[0, :]

    for t in xrange(1, length):
        for i in xrange(height):
            min_cost = float('+inf')
            min_j = None
            for j in xrange(height):
                cost = costs[t - 1, j] + output[t, i] / model[i, j]
                if cost < min_cost:
                    min_cost = cost
                    min_j = j
            costs[t, i] = min_cost
            prev[t, i] = min_j

    path = np.zeros(length)
    path[-1] = np.argmin(costs[-1, :])
    for t in xrange(length - 1, 0, -1):
        path[t] = prev[t, path[t + 1]]

    return path
