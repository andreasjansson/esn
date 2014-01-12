import numpy as np
import scipy.sparse

from esn import selforganising

import matplotlib.pyplot as plt
plt.ion()

def test_data(sequence_length=2000, min_freq=20, max_freq=100, sr=1000):

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

    return audio, outputs

n_inputs = 1
n_outputs = 2
width = 50
height = 1
n_internal = width * height

network = selforganising.SelfOrganisingReservoir(
    n_inputs=n_inputs,
    n_outputs=n_outputs,
    width=width,
    height=height,
    input_scaling=10,
    internal_scaling=2,
    leakage=.2,
    learning_rate_start=.05,
    neighbourhood_width_start=5,
)

inputs, outputs = test_data()

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

#network.internal_weights = _generate_internal_weights()
network.pretrain(inputs)

network.fit(inputs, outputs)
predicted = network.predict(inputs, outputs)

plt.imshow(network.internal_weights, interpolation='none')

import ipdb; ipdb.set_trace()
