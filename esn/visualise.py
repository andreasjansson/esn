import cPickle
import wx
import math
import collections
import time
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import cPickle

from esn import NeighbourESN, nrmse, mean_error, OnlineNeighbourESN
#import test_data

app = None

class Visualiser(wx.Frame):

    def __init__(self, network, refresh_rate=1, input_yscale=1, internal_yscale=1, output_yscale=1):
        global app
        if app is None:
            app = wx.App()

        super(Visualiser, self).__init__(None, -1, 'esn', style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
        #self.SetDoubleBuffered(True)
        self.SetSizeHints(1500, 800)

        self.network = network
        self.input_neurons = {}
        self.internal_neurons = {}
        self.output_neurons = {}
        self.internal_synapses = {}

        self.input_yscale = input_yscale
        self.internal_yscale = internal_yscale
        self.output_yscale = output_yscale

        self.set_refresh_rate(refresh_rate)

        self.main_panel = wx.Panel(self)
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.main_panel.SetSizer(self.main_sizer)

        self.add_input_neurons()
        self.add_internal_neurons()
        self.add_output_neurons()

        self.set_weights()

        self.Update()
        self.Show()

    def set_refresh_rate(self, refresh_rate):
        if refresh_rate:
            self.network.callback = self.refresh
            self.network.callback_every = refresh_rate
        else:
            self.network.callback = None
            self.network.callback_every = refresh_rate

    def add_input_neurons(self):
        panel = wx.Panel(self.main_panel)
        self.main_sizer.Add(panel, 1, wx.EXPAND)

        cols = self.network.n_input_units * 2 - 1
        sizer = wx.GridSizer(1, cols)
        panel.SetSizer(sizer)

        for col in range(cols):
            if col % 2 == 0:
                neuron = Neuron(panel, yscale=self.input_yscale)
                self.input_neurons[col / 2] = neuron
                sizer.Add(neuron, flag=wx.EXPAND)
            else:
                sizer.Add(wx.Panel(panel), 0, wx.EXPAND)

    def add_output_neurons(self):
        cols = self.network.n_output_units * 2 - 1

        panel = wx.Panel(self.main_panel)
        self.main_sizer.Add(panel, 1, wx.EXPAND)

        sizer = wx.GridSizer(1, cols)
        panel.SetSizer(sizer)

        for col in range(cols):
            if col % 2 == 0:
                neuron = Neuron(panel, yscale=self.output_yscale)
                self.output_neurons[col / 2] = neuron
                sizer.Add(neuron, flag=wx.EXPAND)
            else:
                sizer.Add(wx.Panel(panel), 0, wx.EXPAND)

    def add_internal_neurons(self):
        panel = wx.Panel(self.main_panel)
        self.main_sizer.Add(panel, 8, wx.EXPAND)

        cols = self.network.width * 2 - 1
        rows = self.network.height * 2 - 1
        sizer = wx.GridSizer(rows, cols)

        panel.SetSizer(sizer)

        for row in range(rows):
            if row % 2 == 0:
                y = row / 2

            for col in range(cols):
                if col % 2 == 0:
                    x = col / 2

                if col % 2 == 0 and row % 2 == 0:
                    neuron = Neuron(panel, yscale=self.internal_yscale)
                    self.internal_neurons[(x, y)] = neuron
                    sizer.Add(neuron, flag=wx.EXPAND)

                elif isinstance(self.network, NeighbourESN):
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
                        weight = self.network.get_internal_weight(x1, y1, x2, y2)
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

                else:
                    sizer.Add(wx.Panel(panel))
                

    def set_weights(self):

        for ((x1, y1), (x2, y2)), synapse_group in self.internal_synapses.iteritems():
            weight = self.network.get_internal_weight(x1, y1, x2, y2)
            synapse_group.set_weight(((x1, y1), (x2, y2)), weight)

        for (x, y), neuron in self.internal_neurons.iteritems():
            weight = self.network.get_internal_to_output_weight(0, x, y)
            neuron.weight = weight

        for i, neuron in self.input_neurons.iteritems():
            weight = self.network.get_input_to_output_weight(0, i)
            neuron.weight = weight

    def refresh(self, data, actual_output=None):

        for i, neuron in self.input_neurons.iteritems():
            neuron.set_activations(data[:, i])

        for (x, y), neuron in self.internal_neurons.iteritems():
            neuron.set_activations(data[:, self.network.n_input_units + self.network.point_to_index(x, y)])
            
        for i, neuron in self.output_neurons.iteritems():
            activations = data[:, self.network.n_input_units + self.network.n_internal_units + i]
            if actual_output is not None:
                neuron.set_activations(activations, actual_output[:, i])
            else:
                neuron.set_activations(activations)

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
        self.secondary_history = None
        self.weight = 0
        self.yscale = yscale

    def set_activation(self, activation):
        self.history.append(activation)
        self.dirty = True
#        self.Refresh()

    def set_activations(self, history, secondary_history=None):
        if secondary_history is not None:
            if self.secondary_history is None:
                self.secondary_history = collections.deque(maxlen=self.history_length)
            self.secondary_history.extend(secondary_history[-self.history_length:])

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

        yscale = - min(w, h) * self.yscale
        yshift = h / 2

        dc.SetPen(wx.Pen(wx.RED))

        if self.secondary_history is not None:
            for i in xrange(len(self.history) - 1):
                x1 = w * i / float(self.history_length)
                y1 = self.secondary_history[i] * yscale + yshift
                x2 = w * (i + 1) / float(self.history_length)
                y2 = self.secondary_history[i + 1] * yscale + yshift
                dc.DrawLine(x1, y1, x2, y2)
            
        dc.SetPen(wx.Pen(wx.BLACK))

        for i in xrange(len(self.history) - 1):
            x1 = w * i / float(self.history_length)
            y1 = self.history[i] * yscale + yshift
            x2 = w * (i + 1) / float(self.history_length)
            y2 = self.history[i + 1] * yscale + yshift
            dc.DrawLine(x1, y1, x2, y2)


visualiser = None
esn = None

def refresh(data=None, actual_output=None):
    while app.Pending():
        app.Dispatch()
    if data is not None:
        visualiser.on_training_update(data, actual_output)
        visualiser.set_weights()

last_time = None
def refresh_midi(data=None):
    global last_time

    import scikits.audiolab
    import alsaseq, alsamidi
    
    this_time = time.time()
    if last_time is None:
        last_time = this_time
    sleep = max(0, .2 - (this_time - last_time))

    print sleep

    while app.Pending():
        app.Dispatch()
    if data is not None:
        visualiser.on_training_update(data)
        out = data[-1, -esn.n_output_units:]
        out = (out - esn.teacher_shift) / esn.teacher_scaling
        notes_to_output = collections.defaultdict(list)
        for i, x in enumerate(out):
            # if i > 18:
            #     chan = 2
            #     note = {19: 29, 20: 31, 21: 33}[i]
            # elif i > 15:
            #     chan = 1
            #     note = {16: 36, 17: 40, 18: 44}[i]
            # else:
            #     chan = 0
            #     note = i + 69 - 24
            chan = 0
            note = i + 40
            alsaseq.output(alsamidi.noteoffevent(chan, note, 100))
            if x > .5:
                notes_to_output[chan].append((x, note))
        for chan, notes in notes_to_output.items():
            if len(notes) > 3:
                notes = sorted(notes)[:3]
            for _, note in notes:
                alsaseq.output(alsamidi.noteonevent(chan, note, 100))
        drawing_time = time.time() - this_time
        time.sleep(sleep)

    last_time = time.time() - drawing_time

def music():
    global esn, visualiser

    train_skip = sys.argv[1]
    test_skip = sys.argv[2]

    input, output, esn = test_data.bach()
    n_forget_points = 0

    alsaseq.client('andreas', 1, 1, True)
    alsaseq.connectto(1, 20, 0)
    alsaseq.start()

    visualiser = Visualiser(esn)

    if len(sys.argv) < 4:
        state_matrix = esn.train(input, output, n_forget_points=n_forget_points, callback_every=int(train_skip), callback=refresh_midi)

    else:
        with open(sys.argv[3]) as f:
            esn.unserialize(cPickle.load(f))

    visualiser.set_weights()
    esn.reset_state()

    esn.noise_level = 0

    print 'test'
    estimated_output = esn.test(input, callback=refresh_midi, n_forget_points=n_forget_points, callback_every=int(test_skip))

    error = nrmse(estimated_output, output)
    print 'error: %s' % error


def instrumentalness():
    global esn, visualiser

    train_skip = sys.argv[1]
    test_skip = sys.argv[2]
    pkl = sys.argv[3] if len(sys.argv) > 3 else None
    n_forget_points = 0

    train_input, train_output, train_splits, test_input, test_output, test_splits, esn = test_data.instrumentalness()

    visualiser = Visualiser(esn, output_yscale=.3)

    if pkl:
        with open(pkl) as f:
            esn.unserialize(cPickle.load(f))
    else:
        esn.train(train_input, train_output, callback=refresh, n_forget_points=n_forget_points, callback_every=int(train_skip), reset_points=train_splits)

    visualiser.set_weights()
    esn.reset_state()

    esn.noise_level = 0

    #test_input, test_output, test_splits = train_input, train_output, train_splits

    import ipdb; ipdb.set_trace()

    print 'test'
    estimated_output = esn.test(test_input, n_forget_points=n_forget_points, callback_every=int(test_skip), callback=refresh, reset_points=test_splits, actual_output=test_output * esn.teacher_scaling + esn.teacher_shift)

    error = nrmse(estimated_output, test_output)
    print 'error: %s' % error

    means = []
    estimated_means = []
    for start, end in zip(test_splits[:-1], test_splits[1:]):
        means.append(np.mean(test_output[start:end]))
        estimated_means.append(np.mean(estimated_output[start:end]))

    plt.plot(means, 'go', markersize=10)
    plt.plot(estimated_means, 'r*', markersize=10)

    for i, (out, est) in enumerate(zip(means, estimated_means)):
        if out > est:
            col = 'c'
        else:
            col = 'y'
            
        i /= float(len(means))
        plt.axhspan(out, est, i - .005, i + 0.005, color=col)

    plt.show()

def generic():
    global esn, visualiser

    train_skip = sys.argv[1]
    test_skip = sys.argv[2]
    n_forget_points = 0

    input, output, esn = test_data.scholarpedia()

    visualiser = Visualiser(esn)

    state_matrix = esn.train(input, output, callback=refresh, n_forget_points=n_forget_points,
                             callback_every=int(train_skip))

    visualiser.set_weights()
    esn.reset_state()

    esn.noise_level = 0

    print 'test'
    estimated_output = esn.test(input, n_forget_points=n_forget_points, callback_every=int(test_skip), callback=refresh)

    error = nrmse(estimated_output, output)
    print 'error: %s' % error

    plt.plot(output[n_forget_points:])
    plt.plot(estimated_output)

    plt.show()


if __name__ == '__main__':
    # waveform=sin
    # train_skip=400
    # test_skip=400

    instrumentalness()
    #music()
    #generic()


#    scikits.audiolab.play(estimated_output.T / max(abs(estimated_output)), fs=test_data.SR)
