import cPickle
import wx
import math
import collections
import threading
import random
import matplotlib.pyplot as plt
import numpy as np
import scikits.audiolab

from esn import NeighbourESN, optimise, nrmse

callback_event_type = wx.NewEventType()
EVT_CALLBACK_EVENT = wx.PyEventBinder(callback_event_type, 1)

class CallbackEvent(wx.PyCommandEvent):
    def __init__(self, evt_type, id, data):
        wx.PyCommandEvent.__init__(self, evt_type, id)
        self.data = data

SR = 6000

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

    return pitches, audio

def test_data3(sequence_length=8000, min_pitch=30, max_pitch=60, sr=SR):

    pitches = np.zeros((sequence_length, 1))
    audio1 = np.zeros((sequence_length, 1))
    audio2 = np.zeros((sequence_length, 1))

    notes = collections.deque([48, 51, 55, 60, 59, 60])

    current_pitch = pitches[0]

    for i in xrange(sequence_length):
        if i % 2000 == 0:
            notes.rotate(-1)
            current_pitch = notes[0]

        pitches[i, 0] = current_pitch

    phase = 0
    c4 = 261.63
    for i in xrange(1, sequence_length):
        freq = c4 * np.power(2.0, (pitches[i, 0] - 60) / 12.0)
        phase += 2 * np.pi * freq / sr
        if phase > 2 * np.pi:
            phase -= 2 * np.pi
        audio1[i, 0] = np.sin(phase)
        audio2[i, 0] = np.int(phase / np.pi)

    audio1 = (audio1 + 1) / 2
    audio2 = (audio2 + 1) / 2

    return audio1, audio2

class Visualiser(wx.Frame):

    def __init__(self, neighbour_esn):
        super(Visualiser, self).__init__(None, -1, 'esn',
        style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
        #self.SetDoubleBuffered(True)
        self.SetSizeHints(800, 800)

        self.esn = neighbour_esn
        self.input_neurons = {}
        self.internal_neurons = {}
        self.output_neurons = {}
        self.internal_synapses = {}

        self.main_panel = wx.Panel(self)
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.main_panel.SetSizer(self.main_sizer)

        self.add_input_neurons()
        self.add_internal_neurons()
        self.add_output_neurons()

        self.set_weights()

        self.Bind(EVT_CALLBACK_EVENT, self.on_training_update)

    def add_input_neurons(self):
        panel = wx.Panel(self.main_panel)
        self.main_sizer.Add(panel, 1, wx.EXPAND)

        cols = self.esn.n_input_units * 2 - 1
        sizer = wx.GridSizer(cols, 1)
        panel.SetSizer(sizer)

        for col in range(cols):
            if col % 2 == 0 and col == 0:
                neuron = Neuron(panel, yscale=.5, history_length=300 / cols)
                self.input_neurons[col / 2] = neuron
                sizer.Add(neuron, flag=wx.EXPAND)
#            else:
#                sizer.Add(wx.StaticText(self, -1, ''), 0, wx.EXPAND)

    def add_output_neurons(self):
        cols = self.esn.n_output_units * 2 - 1

        panel = wx.Panel(self.main_panel)
        self.main_sizer.Add(panel, 1, wx.EXPAND)

        sizer = wx.GridSizer(cols, 1)
        panel.SetSizer(sizer)

        for col in range(cols):
            if col % 2 == 0:
                neuron = Neuron(panel, yscale=0.3, history_length=300 / cols)
                self.output_neurons[col / 2] = neuron
                sizer.Add(neuron, flag=wx.EXPAND)

    def add_internal_neurons(self):
        panel = wx.Panel(self.main_panel)
        self.main_sizer.Add(panel, 8, wx.EXPAND)

        cols = self.esn.width * 2 - 1
        rows = self.esn.height * 2 - 1
        sizer = wx.GridSizer(cols, rows)

        for row in range(rows):
            if row % 2 == 0:
                y = row / 2

            for col in range(cols):
                if col % 2 == 0:
                    x = col / 2

                if col % 2 == 0 and row % 2 == 0:
                    neuron = Neuron(panel)
                    self.internal_neurons[(x, y)] = neuron
                    sizer.Add(neuron, flag=wx.EXPAND)

                else:
                    if col % 2 == 1 and row % 2 == 1:
                        x1 = col // 2
                        x2 = x1 + 1
                        y1 = row // 2
                        y2 = y1 + 1
                        top_left = (x1, y1)
                        top_right = (x2, y1)
                        bottom_left = (x1, y2)
                        bottom_right = (x2, y2)
                        directions = [(top_left, bottom_right), (top_right, bottom_left)]

                    elif col % 2 == 0:
                        x = col / 2
                        y1 = row // 2
                        y2 = y1 + 1
                        top = (x, y1)
                        bottom = (x, y2)
                        directions = [(top, bottom)]

                    elif row % 2 == 0:
                        x1 = col // 2
                        x2 = x1 + 1
                        y = row / 2
                        left = (x1, y)
                        right = (x2, y)
                        directions = [(left, right)]

                    real_directions = []
                    for ((x1, y1), (x2, y2)) in directions:
                        weight = self.esn.get_internal_weight(x1, y1, x2, y2)
                        if weight == 0:
                            x1, y1, x2, y2 = x2, y2, x1, y1
                        real_directions.append(((x1, y1), (x2, y2)))
                    positions = {}
                    for direction in real_directions:
                        (x1, y1), (x2, y2) = direction
                        if x1 == x2 and y1 < y2:
                            positions[direction] = ((.5, 0), (.5, 1))
                        elif x1 == x2 and y1 > y2:
                            positions[direction] = ((.5, 1), (.5, 0))
                        elif y1 == y2 and x1 < x2:
                            positions[direction] = ((0, .5), (1, .5))
                        elif y1 == y2 and x1 > x2:
                            positions[direction] = ((1, .5), (0, .5))
                        elif x1 < x2 and y1 < y2:
                            positions[direction] = ((0, 0), (1, 1))
                        elif x1 < x2 and y1 > y2:
                            positions[direction] = ((0, 1), (1, 0))
                        elif x1 > x2 and y1 < y2:
                            positions[direction] = ((1, 0), (0, 1))
                        elif x1 > x2 and y1 > y2:
                            positions[direction] = ((1, 1), (0, 0))
                            
                    synapse_group = SynapseGroup(panel, positions)
                    for direction in real_directions:
                        self.internal_synapses[direction] = synapse_group
                        
                    sizer.Add(synapse_group, flag=wx.EXPAND)

        panel.SetSizer(sizer)

    def set_weights(self):

        for ((x1, y1), (x2, y2)), synapse_group in self.internal_synapses.iteritems():
            weight = self.esn.get_internal_weight(x1, y1, x2, y2)
            synapse_group.set_weight(((x1, y1), (x2, y2)), weight)

        for (x, y), neuron in self.internal_neurons.iteritems():
            weight = self.esn.get_output_weight(0, x, y)
            neuron.weight = weight

    def on_training_update(self, data):

        for i, neuron in self.input_neurons.iteritems():
            neuron.set_activations(data[:, i])

        for (x, y), neuron in self.internal_neurons.iteritems():
            neuron.set_activations(data[:, self.esn.n_input_units + self.esn.point_to_index(x, y)])
            
        for i, neuron in self.output_neurons.iteritems():
            neuron.set_activations(data[:, self.esn.n_input_units + self.esn.n_internal_units + i])

        self.Update()

class Sprite(wx.Panel):

    def __init__(self, parent):
        super(Sprite, self).__init__(parent)

        self.dirty = True

        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def on_size(self, event):
        event.Skip()
        self.dirty = True
        self.Refresh()

    def on_paint(self, event):
        if self.dirty:
            self.repaint()
            self.dirty = False

    def repaint(self):
        pass

class SynapseGroup(Sprite):

    def __init__(self, parent, positions):
        super(SynapseGroup, self).__init__(parent)
        self.weights = {d: 0 for d in positions.keys()}
        self.positions = positions

    def set_weight(self, direction, weight):
        if weight != self.weights[direction]:
            self.weights[direction] = weight
            self.dirty = True

    def repaint(self):
        w, h = self.GetClientSize()
        dc = wx.AutoBufferedPaintDC(self)
        dc.Clear()

        end_width = 6
        end_height = (math.sqrt(3) / 2) * end_width
        end_y = end_height
        end_x1 = end_width / 2
        end_x2 = -end_width / 2

        for direction, weight in self.weights.iteritems():
            (x1, y1), (x2, y2) = self.positions[direction]
            x1 *= w
            y1 *= h
            x2 *= w
            y2 *= h

            if weight > 0:
                c = 255 - int(min(weight, 1) * 255.0)
                colour = wx.Colour(c, 255, c)
            else:
                c = 255 - int(min(-weight, 1) * 255.0)
                colour = wx.Colour(255, c, c)

            pen = wx.Pen(colour, 2)
            if abs(weight) > 1:
                pen.SetStyle(wx.SHORT_DASH)
            dc.SetPen(pen)

            dc.DrawLine(x1, y1, x2, y2)

            a = math.atan2(x2 - x1, y2 - y1)
            rot_x1 = end_x1 * math.cos(a) + end_y * math.sin(a)
            rot_y1 = -end_x1 * math.sin(a) + end_y * math.cos(a)
            rot_x2 = end_x2 * math.cos(a) + end_y * math.sin(a)
            rot_y2 = -end_x2 * math.sin(a) + end_y * math.cos(a)

            dc.DrawPolygon([(x2, y2),
                            (x2 - rot_x1, y2 - rot_y1),
                            (x2 - rot_x2, y2 - rot_y2)])

class Neuron(Sprite):

    def __init__(self, parent, yscale=.5, history_length=50):
        super(Neuron, self).__init__(parent)

        self.history_length = history_length
        self.history = collections.deque(maxlen=self.history_length)
        self.weight = 0
        self.yscale = yscale

    def set_activation(self, activation):
        self.history.append(activation)
        self.dirty = True
#        self.Refresh()

    def set_activations(self, history):
        self.history.extend(history[-self.history_length:])
        self.repaint()

    def repaint(self):
        w, h = self.GetClientSize()
        dc = wx.AutoBufferedPaintDC(self)

        if self.weight > 0:
            c = 255 - int(min(self.weight, 1) * 255.0)
            colour = wx.Colour(c, 255, c)
        else:
            c = 255 - int(min(-self.weight, 1) * 255.0)
            colour = wx.Colour(255, c, c)

        pen = wx.Pen(colour, 2)
        dc.SetPen(pen)
        dc.DrawRectangle(0, 0, w, h)

        dc.SetPen(wx.Pen(wx.BLACK))

        yscale = - min(w, h) * self.yscale
        yshift = h / 2
        for i in xrange(len(self.history) - 1):
            x1 = w * i / float(self.history_length)
            y1 = self.history[i] * yscale + yshift
            x2 = w * (i + 1) / float(self.history_length)
            y2 = self.history[i + 1] * yscale + yshift
            dc.DrawLine(x1, y1, x2, y2)

def raise_callback_event(callback_state):
    evt = CallbackEvent(callback_event_type, -1, callback_state)
    wx.PostEvent(visualiser, evt)


if __name__ == '__main__':
    # waveform=sin
    # train_skip=400
    # test_skip=400
    import sys
    _, waveform, train_skip, test_skip = sys.argv
    input, output = test_data2(waveform)

    # for i in range(50):
    #     esn = NeighbourESN(
    #         n_input_units=1,
    #         width=13,
    #         height=13,
    #         n_output_units=1,
    #         input_scaling=[1.7 + random.random() * .4 - .2],
    #         input_shift=[-.3 + random.random() * .4 - .2],
    #         teacher_scaling=[0.015 + random.random() * .004 - .002],
    #         teacher_shift=[-.8 + random.random() * .4 - .2],
    #         noise_level=0.001 + random.random() * .004 - .002,
    #         spectral_radius=1.01 + random.random() * .02 - .01,
    #         feedback_scaling=[.8 + random.random() * .4 - .2],
    #     #    feedback_scaling=[1000],
    #         output_activation_function='tanh',
    #     )



    #     esn, estimated_output, error = optimise(esn, input, output, 1000, 30)
    #     print error

    #     with open('network-%s.pkl' % error, 'w') as f:
    #         cPickle.dump(esn, f)

    #import sys
    #with open(sys.argv[1], 'r') as f:
    #    esn = cPickle.load(f)

    # esn = NeighbourESN(
    #     n_input_units=1,
    #     width=8,
    #     height=8,
    #     n_output_units=1,
    #     input_scaling=[1.7 + random.random() * .4 - .2],
    #     input_shift=[-.3 + random.random() * .4 - .2],
    #     teacher_scaling=[0.015 + random.random() * .004 - .002],
    #     teacher_shift=[-.8 + random.random() * .4 - .2],
    #     noise_level=0.001 + random.random() * .004 - .002,
    #     spectral_radius=1.01 + random.random() * .02 - .01,
    #     feedback_scaling=[.8 + random.random() * .4 - .2],
    # #    feedback_scaling=[1000],
    #     output_activation_function='tanh',
    # )

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

    app = wx.App()
    visualiser = Visualiser(esn)

    visualiser.Show()

    visualiser.Update()

    def refresh(data=None):
        while app.Pending():
            app.Dispatch()
        if data is not None:
            visualiser.on_training_update(data)

    state_matrix = esn.train(input, output, callback=refresh, n_forget_points=1000, callback_every=int(train_skip))
    visualiser.set_weights()
    esn.reset_state()
    esn.noise_level = 0
    print 'test'
    estimated_output = esn.test(input, callback=refresh, n_forget_points=1000, callback_every=int(test_skip))

    error = np.sum(nrmse(estimated_output, output))
    print 'error: %s' % error

    visualiser.Close()
    refresh()

    scikits.audiolab.play(estimated_output.T / max(abs(estimated_output)), fs=SR)

    plt.plot(output[1000:])
    plt.plot(estimated_output)
    plt.show()
