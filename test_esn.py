from esn import EchoStateNetwork, optimise, NeighbourESN, GeneticOptimiser, fmin
import unittest2 as unittest
import numpy as np
import matplotlib.pyplot as plt
import minimidi
import pyaudio
import fluidsynth
import struct

SR = 4000

def scholarpedia_esn():
    return EchoStateNetwork(
        n_input_units=2,
        n_internal_units=200,
        n_output_units=1,
        connectivity=0.05,
        input_scaling=[0.01, 3],
        input_shift=[0, 0],
        teacher_scaling=[1.4],
        teacher_shift=[-0.7],
        noise_level=0.001,
        spectral_radius=0.25,
        feedback_scaling=[0.8],
    )

def scholarpedia_data(sequence_length=5000, out_min_period=4, out_max_period=16):
    out_period_setting = np.zeros((sequence_length, 1))
    current_value = np.random.rand()
    for i in xrange(sequence_length):
        if np.random.rand() < 0.015:
            current_value = np.random.rand()
        out_period_setting[i, 0] = current_value

    ones = np.ones(sequence_length).reshape((sequence_length, 1))
    input = np.hstack((ones, 1 - out_period_setting))

    current_sin_arg = 0
    output = np.zeros((sequence_length, 1))
    for i in xrange(1, sequence_length):
        current_out_period_length = out_period_setting[i-1, 0] * (out_max_period - out_min_period) + out_min_period
        current_sin_arg = current_sin_arg + 2 * np.pi / current_out_period_length
        output[i, 0] = (np.sin(current_sin_arg) + 1) / 2

    return input, output


def my_multiclass_esn():
    n_out = 3
    return EchoStateNetwork(
        n_input_units=1,
        n_internal_units=200,
        n_output_units=n_out,
        connectivity=0.1,
        input_scaling=[3],
        input_shift=[0],
        teacher_scaling=[.4] * n_out,
        teacher_shift=[-0.5] * n_out,
        noise_level=0.001,
        spectral_radius=0.85,
        feedback_scaling=[1.3] * n_out,
        output_activation_function='tanh',
    )

def my_multiclass_data(sequence_length=10000, out_min_period=1, out_max_period=5):
    out_period_setting = np.zeros((sequence_length, 1))
    #current_value = np.random.rand()

    output = np.zeros((sequence_length, 3))

    current_value = .1
    for i in xrange(sequence_length):
#        if np.random.rand() < 0.0015:
#            current_value = np.random.rand()

        if i % 2000 == 0:
            current_value += 1
            current_value = current_value % 2

        out_period_setting[i, 0] = current_value
        output[i, current_value] = 1.0

    current_sin_arg = 0
    input = np.zeros((sequence_length, 1))
    for i in xrange(1, sequence_length):
        current_out_period_length = out_period_setting[i-1, 0] * (out_max_period - out_min_period) + out_min_period
        current_sin_arg = current_sin_arg + 2 * np.pi / current_out_period_length
        input[i, 0] = (np.sin(current_sin_arg) + 1) / 2

    return input, output

def my_single_class_esn():
    n_out = 1
    return EchoStateNetwork(
        n_input_units=1,
        n_internal_units=100,
        n_output_units=n_out,
        connectivity=0.1,
        input_scaling=[3],
        input_shift=[0],
        teacher_scaling=[0.001] * n_out,
        teacher_shift=[-.05] * n_out,
        noise_level=0.001,
        spectral_radius=0.85,
        feedback_scaling=[1.3] * n_out,
        output_activation_function='tanh',
    )

def my_single_class_data(sequence_length=5000, min_freq=20, max_freq=100, sr=SR):

    freqs = np.zeros((sequence_length, 1))
    audio = np.zeros((sequence_length, 1))

    def rnd_freq():
        return np.random.rand() * (max_freq - min_freq) + min_freq

    current_freq = rnd_freq()

    for i in xrange(sequence_length):
        if np.random.rand() < 0.03:
            current_freq = rnd_freq()

        freqs[i, 0] = current_freq

    phase = 0
    for i in xrange(1, sequence_length):
        freq = freqs[i, 0]
        phase += 2 * np.pi / sr
        if phase > 2 * np.pi:
            phase -= 2 * np.pi
        audio[i, 0] = np.sin(phase * freq)

#        if i > 1000:
#            audio[i, 0] = 0

    audio = (audio + 1) / 2

    return audio, freqs

def my_pitched_data(sequence_length=10000, min_pitch=60, max_pitch=71, sr=SR):

    pitches = np.zeros((sequence_length, 1))
    audio = np.zeros((sequence_length, 1))

    def rnd_pitch():
        return round(np.random.rand() * (max_pitch - min_pitch) + min_pitch)

    current_pitch = rnd_pitch()
    current_pitch = 60

    for i in xrange(sequence_length):
        pitches[i, 0] = current_pitch

        if i % 1000 == 0:
            current_pitch += 1

    phase = 0
    c4 = 261.63
    for i in xrange(1, sequence_length):
        freq = c4 * np.power(2.0, (pitches[i, 0] - 60) / 12.0)
        phase += 2 * np.pi / sr
        if phase > 2 * np.pi:
            phase -= 2 * np.pi
        audio[i, 0] = np.sin(phase * freq)

    audio = (audio + 1) / 2

    return pitches, audio

def my_piano_data(note_length=4000, min_pitch=60, max_pitch=64, sr=SR):

    synth = fluidsynth.Synth(samplerate=SR)
    soundfont = synth.sfload('/usr/share/soundfonts/fluidr3/FluidR3GM.SF2')
    synth.program_select(0, soundfont, 0, 41)

    n_notes = max_pitch - min_pitch
    pitches = np.zeros((note_length * n_notes, 1))
    audio = np.zeros((note_length * n_notes, 1))

    t = 0
    for pitch in range(min_pitch, max_pitch):
        synth.noteon(0, pitch, 127)
        audio[t : t + note_length, 0] = synth.get_samples(note_length)[::2]
        pitches[t : t + note_length, 0] = [pitch] * note_length
        synth.noteoff(0, pitch)
        t += note_length

    audio = audio / 3200.0

    return pitches, audio

def scholarpedia_data2(sequence_length=5000, out_min_period=4, out_max_period=16):
    out_period_setting = np.zeros((sequence_length, 1))
    current_value = np.random.rand()
    for i in xrange(sequence_length):
        if np.random.rand() < 0.015:
            current_value = np.random.rand()
        out_period_setting[i, 0] = current_value

    input = out_period_setting

    current_sin_arg = 0
    output = np.zeros((sequence_length, 1))
    for i in xrange(1, sequence_length):
        current_out_period_length = out_period_setting[i-1, 0] * (out_max_period - out_min_period) + out_min_period
        current_sin_arg = current_sin_arg + 2 * np.pi / current_out_period_length
        output[i, 0] = (np.sin(current_sin_arg) + 1) / 2

    return input, output

def my_midi_data():

    pitches = [0, 0, 0, 1, 0, 0, 1, 0, 1] * 100
    output = np.zeros((len(pitches), 2))
    for i, pitch in enumerate(pitches):
        output[i, pitch] = 1

    input = np.ones((len(pitches), 1))

    return input, output

def split_data(input, output, train_fraction=0.8):
    train_length = len(input) * train_fraction
    return input[:train_length], output[:train_length], input[train_length:], output[train_length:]

def nrmse(estimated, correct):
    n_forget_points = len(correct) - len(estimated)
    correct = correct[n_forget_points:, :]
    correct_variance = np.var(correct)
    mean_error = sum(np.power(estimated - correct, 2)) / len(estimated)
    return np.sqrt(mean_error / correct_variance)

class TestEchoStateNetwork(unittest.TestCase):

    def test_init(self):
        esn = scholarpedia_esn()
        self.assertEquals((200, 200), np.shape(esn.internal_weights))
        self.assertAlmostEqual(0.25, np.max(np.abs(np.linalg.eigvals(esn.internal_weights))))
        n_nonzeros = np.sum(esn.internal_weights != 0)
        self.assertEquals(0.05 * 200 * 200, n_nonzeros)

    def test_train(self):
        input, output = scholarpedia_data()
        train_input, train_output, test_input, test_output = split_data(input, output)

        esn = scholarpedia_esn()
        state_matrix = esn.train(train_input, train_output, 100)

        predicted_train_output = state_matrix.dot(esn.output_weights.T)
        predicted_train_output = esn.output_activation_function(predicted_train_output)
        predicted_train_output -= esn.teacher_shift
        predicted_train_output /= esn.teacher_scaling
        self.assertLess(nrmse(predicted_train_output, train_output), 0.1)

        output = esn.test(train_input)
        plt.plot(train_output)
        plt.plot(output)
        plt.show()

    def test_my_esn(self):
        train_input, train_output = my_data()
        esn = my_esn()
        esn.train(train_input, train_output, 1000)

#        esn.noise_level = 0

        output = esn.test(train_input)
        plt.plot(train_output)
        plt.plot(output)
        plt.show()
        
class TestEvaluate(unittest.TestCase):

    def test_evaluate(self):
        input, output = my_single_class_data()
        esn = my_single_class_esn()
        estimated_output, error = evaluate(esn, input, output, 1000)
        import ipdb; ipdb.set_trace()

current_pitch = 60

class TestNeighbourESN(unittest.TestCase):

    def test_neighbour_esn(self):
        input, output = my_single_class_data()

        esn = NeighbourESN(
            n_input_units=1,
            n_internal_units=64,
            n_output_units=1,
            input_scaling=[3],
            input_shift=[0],
            teacher_scaling=[0.001],
            teacher_shift=[-.05],
            noise_level=0.001,
            spectral_radius=0.85,
            feedback_scaling=[1.3],
            output_activation_function='tanh',
        )
        estimated_output, error = evaluate(esn, input, output, 1000, 5)
        import ipdb; ipdb.set_trace()

    def test_neighbour_esn2(self):
        input, output = my_piano_data()

        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, rate=SR,
                            channels=1, output=True)

        import ipdb; ipdb.set_trace()

        play(stream, output)

        esn = NeighbourESN(
            n_input_units=1,
            n_internal_units=10*10,
            n_output_units=1,
            input_scaling=[0.1],
            input_shift=[0],
            teacher_scaling=[1.4],
            teacher_shift=[-0.7],
            noise_level=0.001,
            spectral_radius=0.5,
            feedback_scaling=[0.8],
        )

        esn = EchoStateNetwork(
            n_input_units=1,
            n_internal_units=200,
            n_output_units=1,
            connectivity=0.05,
            input_scaling=[0.01],
            input_shift=[0],
            teacher_scaling=[1.4],
            teacher_shift=[-0.7],
            noise_level=0.001,
            spectral_radius=0.25,
            feedback_scaling=[0.8],
        )

        best_esn, estimated_output, error = optimise(esn, input, output, 500, 5)

        import ipdb; ipdb.set_trace()

        def play_note(channel, pitch, velocity):
            global current_pitch
            print pitch
            current_pitch = pitch

#        minimidi.add_event_listener('note_on', play_note)

        global current_pitch

        play(stream, estimated_output)

        old_pitch = None
        while True:
            if current_pitch != old_pitch:
                print 'pitch: %s' % current_pitch
                output = esn.test(np.array([current_pitch] * 5000).reshape((5000, 1)))
                play(stream, output)
                old_pitch = current_pitch

            current_pitch += 1

        minimidi.remove_all_event_listeners()

class TestOptimise(unittest.TestCase):

    def test_optimise(self):
        import visualise
        import cma

        input, output = visualise.test_data2()

        input_range = np.max(input) + np.min(input)
        input_median = -np.median(input)

        output_range = np.max(output) + np.min(output)
        output_median = -np.median(output)

        esn = NeighbourESN(
        n_input_units=1,
        width=6,
        height=6,
        n_output_units=1,
        input_scaling=[.08],
        input_shift=[-4.4],
        teacher_scaling=[.5],
        teacher_shift=[-.25],
        noise_level=0.002,
        spectral_radius=1.0,
        feedback_scaling=[0.8],
        output_activation_function='identity',
        )

        optimiser = GeneticOptimiser(esn, input, output, 50)
        params = np.array(optimiser.initial_params())
        res = cma.fmin(optimiser.evaluate, params, 0.1)
        #res = fmin(optimiser.evaluate, params)
        import ipdb; ipdb.set_trace()


def play(stream, output):
    output = output - (np.max(output) + np.min(output)) / 2.0
    output *= (2**14 - 1.0) / np.max(output)
    output = output.astype(np.int16).reshape(len(output)).tolist()
    packed_output = struct.pack('%sh' % len(output), *output)
    stream.write(packed_output)

