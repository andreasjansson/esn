import numpy as np

def sequences_from_output(output, split_points):
    sequences = []
    split_points = [0] + split_points
    for t0, t1 in zip(split_points[:-1], split_points[1:]):
        sequence = np.argmax(output[t0:t1, :], 1)
        sequences.append(sequence)
    return sequences

def get_transition_model(sequences, height=26):
    # TODO: take into account previous chord as well as an unordered
    # set of the previous chords over the past n time steps
    model = np.ones((height, height)) # laplace smoothing
    for seq in sequences:
        for x, y in zip(seq[:-1], seq[1:]):
            model[x, y] += 1.0
    model = (model.T / model.sum(1)).T
    return model

def find_sequence(output, model, split_points):
    length, height = output.shape

    costs = np.zeros((length, height))
    prev = np.zeros((length, height))
    costs[0, :] = output[0, :]

    for t in xrange(1, length):
        if t % 1000 == 0:
            print t
        for i in xrange(height):
            min_cost = float('+inf')
            min_j = None
            for j in xrange(height):
                if j in split_points:
                    prev_cost = 0
                else:
                    prev_cost = costs[t - 1, j]
                cost = prev_cost - output[t, i] * model[j, i]
                if cost < min_cost:
                    min_cost = cost
                    min_j = j
            costs[t, i] = min_cost
            prev[t, i] = min_j

    path = np.zeros(length)
    path[-1] = np.argmin(costs[-1, :])
    for t in reversed(range(length - 1)):
        path[t] = prev[t, path[t + 1]]

    return path
