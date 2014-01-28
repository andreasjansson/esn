import numpy as np
import collections
import simplejson as json
import itertools
import glob
import random
import os

from esn import EchoStateNetwork

SR = 6000

def scholarpedia(sequence_length=2000, out_min_period=4, out_max_period=16):

    esn = EchoStateNetwork(
        n_input_units=2,
        width=14,
        height=14,
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

    return input, output, esn


def test_data1(sequence_length=10000, min_freq=20, max_freq=100, sr=SR):

    freqs = np.zeros((sequence_length, 1))
    audio = np.zeros((sequence_length, 1))

    def rnd_freq():
        return np.random.rand() * (max_freq - min_freq) + min_freq

    current_freq = rnd_freq()

    for i in xrange(sequence_length):
        if np.random.rand() < 0.003:
            current_freq = rnd_freq()

        freqs[i, 0] = current_freq

    phase = 0
    for i in xrange(1, sequence_length):
        freq = freqs[i, 0]
        phase += 2 * np.pi / sr
        if phase > 2 * np.pi:
            phase -= 2 * np.pi
        audio[i, 0] = np.sin(phase * freq)

    audio = (audio + 1) / 2

    return audio, freqs

def test_data2(waveform, sequence_length=33000 // 4, min_pitch=30, max_pitch=60, sr=SR):

    pitches = np.zeros((sequence_length, 1))
    audio = np.zeros((sequence_length, 1))

    a = 45
    b = 47
    c = 48
    d = 50
    e = 52
    f = 53
    g = 55
    a2 = 57
    b2 = 59
    c2 = 60
    notes = collections.deque([a2, a2, g, a2, e, c, e, a, a, a2, g, a2, e, c, e, a, a,
                               a2, b2, c2, a2, c2, a2, b2, g, b2, g, a2, f, a2, f, a2, a2])

    def rnd_pitch():
        return round(np.random.rand() * (max_pitch - min_pitch) + min_pitch)

    current_pitch = notes[0]

    for i in xrange(sequence_length):
        pitches[i, 0] = current_pitch

        if i > 0 and i % 1000 == 0:
            notes.rotate(-1)
            current_pitch = notes[0]

    phase = 0
    c4 = 261.63
    for i in xrange(1, sequence_length):
        freq = c4 * np.power(2.0, (pitches[i, 0] - 60) / 12.0) * 2
        phase += 2 * np.pi * freq / sr
        if phase > 2 * np.pi:
            phase -= 2 * np.pi
        if waveform.startswith('sin'):
            audio[i, 0] = np.sin(phase) #sine
        elif waveform.startswith('tri'):
            audio[i, 0] = (2 * (np.pi - np.abs(np.pi - phase)) / np.pi) - 1 #triangle
        elif waveform.startswith('sq'):
            audio[i, 0] = np.int(phase / np.pi) * 2 - 1 #square

    audio = (audio + 1) / 2

#    pitches[:,1] = 1.0

#    scikits.audiolab.play(audio.T, fs=SR)

    esn = NeighbourESN(
        n_input_units=1,
        width=7,
        height=7,
        n_output_units=1,
        input_scaling=[.035],
        input_shift=[-2],
        teacher_scaling=[1.2],
        teacher_shift=[-.6],
        noise_level=0.002,
        spectral_radius=1.3,
        feedback_scaling=[.9],
        output_activation_function='identity',
    )

    esn = NeighbourESN(
        n_input_units=1,
        width=7,
        height=7,
        n_output_units=1,
        input_scaling=[.035],
        input_shift=[-2],
        teacher_scaling=[1.2],
        teacher_shift=[-.6],
        noise_level=0.002,
        spectral_radius=0.5,
        feedback_scaling=[.9],
        output_activation_function='identity',
        leakage=.5,
        time_constants=np.ones((7 * 7, 1)),
    )

    return pitches, audio, esn

def test_data3(waveform, sequence_length=10000, min_pitch=30, max_pitch=60, sr=SR):

    pitches = np.zeros((sequence_length, 2))
    audio = np.zeros((sequence_length, 1))

    a1 = 45
    b1 = 47
    c2 = 48
    d2 = 50
    e2 = 52
    f2 = 53
    g2 = 55
    a2 = 57
    b2 = 59
    c3 = 60
    notes = collections.deque([(c2, g2), (d2, b2), (e2, g2), (f2, a2), (g2, b2)])

    current_pitch = notes[0]

    for i in xrange(sequence_length):
        pitches[i, :] = current_pitch

        if i > 0 and i % 2000 == 0:
            notes.rotate(-1)
            current_pitch = notes[0]

    phases = np.zeros((2))
    c4 = 261.63
    for i in xrange(1, sequence_length):
        freqs = c4 * np.power(2.0, (pitches[i, :] - 60) / 12.0) * 2
        phases += 2 * np.pi * freqs / sr
        phases = phases % (np.pi * 2)
        if waveform.startswith('sin'):
            audio[i, 0] = np.sum(np.sin(phases) / 2) #sine
        elif waveform.startswith('tri'):
            audio[i, 0] = (2 * (np.pi - np.abs(np.pi - phase)) / np.pi) - 1 #triangle
        elif waveform.startswith('sq'):
            audio[i, 0] = np.int(phase / np.pi) * 2 - 1 #square

    audio = (audio + 1) / 2

    esn = NeighbourESN(
        n_input_units=2,
        width=7,
        height=7,
        n_output_units=1,
        input_scaling=[.035, 0.035],
        input_shift=[-2, -2],
        teacher_scaling=[1.2],
        teacher_shift=[-.6],
        noise_level=0.002,
        spectral_radius=0.5,
        feedback_scaling=[.9],
        output_activation_function='identity',
        leakage=.5,
        time_constants=np.ones((7 * 7, 1)),
    )

    return pitches, audio, esn


def instrumentalness_spec(n_pretrain, n_train, n_test, sr=22050, window_size=128, downsample=4):
    spec_dir = '/home/andreas/data/instrumentalness/spectrograms_auditory'
    cache_dir = '/home/andreas/data/instrumentalness/data_cache'
    annotations_dir = '/home/andreas/data/instrumentalness/annotations'

    spec_filenames = sorted(glob.glob('%s/*.mp3_auditory_spectrogram' % spec_dir))

    #random.shuffle(spec_filenames)

    def read_files(spec_filenames):
        split_points = []
        all_data = []
        all_labels = []

        for spec_filename in spec_filenames:
            data = []
            trid = (spec_filename
                    .replace('.mp3_auditory_spectrogram', '.csv')
                    .replace(spec_dir + '/', ''))

            cache_filename = '%s/%s.json' % (cache_dir, trid)
            cached = False
            if os.path.exists(cache_filename):
                with open(cache_filename, 'r') as f:
                    try:
                        data, labels = json.load(f)
                        cached = True
                    except json.scanner.JSONDecodeError:
                        pass

            if not cached:
                annotations_filename = '%s/%s' % (annotations_dir, trid)

                i = 0
                with open(spec_filename, 'r') as f:
                    avg_values = np.zeros(23)
                    for line in f:
                        values = [float(x) for x in line.strip().split(' ')]
                        avg_values += values
                        i += 1
                        if i == downsample:
                            data.append((avg_values / downsample).tolist())
                            avg_values = np.zeros(23)
                            i = 0

                labels = [0] * len(data)

                window_seconds = downsample * window_size / float(sr)
                with open(annotations_filename, 'r') as f:
                    for line in f:
                        split = line.split(',')
                        start = float(split[0])
                        duration = float(split[-1])
                        i = int(start / window_seconds)
                        while i < (start + duration) / window_seconds and i < len(labels):
                            labels[i] = 1
                            i += 1

                with open(cache_filename, 'w') as f:
                    json.dump((data, labels), f)

            all_data += data
            all_labels += labels
            split_points.append(len(all_labels))

        all_labels = np.array(all_labels)
        all_data = np.array(all_data) / 100.
        labels = np.zeros((len(all_labels), 2))
        labels[np.array(all_labels) == 0, 0] = 1
        labels[all_labels == 1, 1] = 1

        return all_data, labels, split_points

    pretrain_inputs, pretrain_outputs, pretrain_split_points = read_files(spec_filenames[:n_pretrain])
    train_inputs, train_outputs, train_split_points = read_files(spec_filenames[n_pretrain:n_pretrain + n_train])
    test_inputs, test_outputs, test_split_points = read_files(spec_filenames[n_pretrain + n_train:n_pretrain + n_train + n_test])

    return pretrain_inputs, pretrain_outputs, pretrain_split_points, train_inputs, train_outputs, train_split_points, test_inputs, test_outputs, test_split_points


def instrumentalness(n_pretrain, n_train, n_test, deterministic=False, side=10):
    segment_dir = '/home/andreas/r/instrumentalness/segments'
    vocal_dir = '/home/andreas/r/instrumentalness/slicing/full_tracks'

    segment_filenames = sorted(glob.glob('%s/*.json' % segment_dir))
    vocal_filenames = sorted(glob.glob('%s/*.csv' % vocal_dir))
    random_ndx = range(len(segment_filenames))

    if not deterministic:
        random.shuffle(random_ndx)

    random_ndx = random_ndx[:n_train + n_test]
    segment_filenames = [segment_filenames[i] for i in random_ndx]
    vocal_filenames = [vocal_filenames[i] for i in random_ndx]

    assert len(segment_filenames) == len(vocal_filenames)

    def get_input_output_splits(segment_filenames, vocal_filenames):
        input = output = None
        splits = [0]
        t = 0
        for segment_filename, vocal_filename in itertools.izip(
                segment_filenames, vocal_filenames):

            with open(segment_filename, 'r') as f:
                segments = json.load(f)

            vocal_segments = []
            with open(vocal_filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    split = line.split(',')
                    start = float(split[0])
                    duration = float(split[-1])
                    vocal_segments.append((start, duration))

            vocal_ndx = 0
            for segment in segments:
                if len(vocal_segments) == 0:
                    segment['vocal'] = False
                    continue

                time = segment['start']
                start, duration = vocal_segments[vocal_ndx]
                while time > start + duration and vocal_ndx < len(vocal_segments) - 1:
                    vocal_ndx += 1
                    start, duration = vocal_segments[vocal_ndx]
                segment['vocal'] = time > start and time < start + duration

            timbres = np.array([s['timbre'] for s in segments])
            vocal = np.array([float(s['vocal']) for s in segments]).reshape((len(segments), 1))

            t += len(segments)

            if input is None:
                input = timbres
                output = vocal
            else:
                input = np.vstack((input, timbres))
                output = np.vstack((output, vocal))
            splits.append(t)

        input /= np.max(input, 0) - np.min(input, 0)
        input -= np.min(input, 0)

        labels = np.zeros((len(output), 2))
        labels[output.ravel() == 0, 0] = 1
        labels[output.ravel() == 1, 1] = 1
        
        return input, labels, splits

    pretrain_inputs, pretrain_outputs, pretrain_split_points = get_input_output_splits(segment_filenames[:n_pretrain], vocal_filenames[:n_pretrain])
    train_inputs, train_outputs, train_split_points = get_input_output_splits(segment_filenames[n_pretrain:n_pretrain + n_train], vocal_filenames[n_pretrain:n_pretrain + n_train])
    test_inputs, test_outputs, test_split_points = get_input_output_splits(segment_filenames[n_pretrain + n_train:n_pretrain + n_train + n_test], vocal_filenames[n_pretrain + n_train:n_pretrain + n_train + n_test])

    return pretrain_inputs, pretrain_outputs, pretrain_split_points, train_inputs, train_outputs, train_split_points, test_inputs, test_outputs, test_split_points


def music():

    a = 0
    b = 2
    c2 = 3
    d2 = 5
    e2 = 7
    f2 = 8
    g2 = 10
    a2 = 12
    b2 = 14
    c3 = 15
    notes1 = [a2, g2, a2, e2,  c2, e2, a, None,  a2, g2, a2, e2,  c2, e2, a, None,
             a2, b2, c3, a2,  c3, a2, b2, g2, b2, g2, a2, f2, a2, f2, a2, None] * 8

    kick = 0
    snare = 1
    hat = 2
    notes2 = [hat, hat, kick, None, hat, None, snare, None] * 8 * 4

    f = 0
    g = 1
    a = 2
    notes3 = [None, None, a, None, None, None, a, None, None, None, a, None, None, None, a, None, 
              None, None, a, None, None, None, g, None, None, None, f, None, None, None, f, None] * 8

    output_units = 16 + 3 + 3
    output = np.zeros((len(notes1), output_units))
    for i, n in enumerate(notes1):
        if n is not None:
            output[i, n] = 1
    for i, n in enumerate(notes2):
        if n is not None:
            output[i, n + 16] = 1
    for i, n in enumerate(notes3):
        if n is not None:
            output[i, n + 16 + 3] = 1

    input_units = 2
    input = np.zeros((len(notes1), 2))
    input[np.arange(len(notes1)) % 32 == 0, 0] = 1
    input[np.arange(len(notes1)) % 32 == 16, 1] = 1

    width = height = 6
    esn = NeighbourESN(
        n_input_units=input_units,
        width=width,
        height=height,
        n_output_units=output_units,
#        input_scaling=[.001],
        input_scaling=[1] * input_units,
        input_shift=[0] * input_units,
        teacher_scaling=[.3] * output_units,
        teacher_shift=[-.1] * output_units,
        noise_level=0.002,
        spectral_radius=1.1,
#        feedback_scaling=[.9] * output_units,
        feedback_scaling=[0] * output_units,
        output_activation_function='identity',
        leakage=1,
        time_constants=np.ones((width * height, 1)),
    )

    esn.input_weights = np.zeros((esn.n_internal_units, esn.n_input_units))
    esn.input_weights[0, 0] = 1
    esn.input_weights[width - 1, 1] = 1

    return input, output, esn

def bach():
    import midi

    m = midi.read_midifile('muss_1.mid')
    m.make_ticks_abs()
    notes = [(n.pitch, int(round(n.tick / 120.0))) for n in m[1] if type(n) == midi.events.NoteOnEvent and n.velocity > 0]

    note_map = collections.defaultdict(list)
    max_pitch = 0
    min_pitch = 127
    for pitch, time in notes:
        note_map[time].append(pitch)
        if pitch > max_pitch:
            max_pitch = pitch
        elif pitch < min_pitch:
            min_pitch = pitch

    input_units = 1

    max_time = max(note_map.keys())
    #max_time = 32 * input_units
    max_time = 140

    output_units = max_pitch - min_pitch + 1
    output = np.zeros((max_time, output_units))
    for t in range(max_time):
        if t in note_map:
            for i in note_map[t]:
                output[t, i - min_pitch] = 1

    input = np.zeros((max_time, input_units))
    for i in range(input_units):
        input[32 * i: 32 * (i + 1), i] = 1
    input = np.zeros((max_time, input_units))

    input[:,0] = np.arange(max_time) / (float(max_time))
    input[:,0] = np.ones((max_time))

    width = height = 8
    esn = NeighbourESN(
        n_input_units=input_units,
        width=width,
        height=height,
        n_output_units=output_units,
#        input_scaling=[.001],
        input_scaling=[1] * input_units,
        input_shift=[-.5] * input_units,
        teacher_scaling=[.2] * output_units,
        teacher_shift=[-.1] * output_units,
        noise_level=0.02,
        spectral_radius=.25,
        feedback_scaling=[.1] * output_units,
#        feedback_scaling=[0] * output_units,
        output_activation_function='tanh',
    )

#    esn.input_weights = np.zeros((esn.n_internal_units, esn.n_input_units))
#    esn.input_weights[0, 0] = 1
#    esn.input_weights[width - 1, 0] = 1

    return input, output, esn
    
