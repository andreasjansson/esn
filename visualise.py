import wx
import math

class Visualiser(wx.Frame):

    def __init__(self, neighbour_esn):
        super(Visualiser, self).__init__(None, -1, 'esn')
#        self.SetDoubleBuffered(True)

        self.esn = neighbour_esn
        self.input_neurons = {}
        self.internal_neurons = {}
        self.output_neurons = {}
        self.arrows = {}

        self.main_panel = wx.Panel(self)
        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.main_panel.SetSizer(self.main_sizer)

        self.add_settings_panel()
        self.add_input_neurons()
        self.add_input_synapses()
        self.add_internal_neurons_and_synapses()
        self.add_feedback_synapses()
        self.add_output_neurons()

        self.set_weights()

    def add_settings_panel(self):
        panel = wx.Panel(self.main_panel)
        panel.SetBackgroundColour('#FFDDDD')
        self.main_sizer.Add(panel, 2, wx.EXPAND | wx.LEFT)

    def add_input_neurons(self):
        panel = wx.Panel(self.main_panel)
        panel.SetBackgroundColour('#FFDDFF')
        self.main_sizer.Add(panel, 1, wx.EXPAND | wx.LEFT)

        sizer = wx.GridSizer(max(5, self.esn.n_input_units * 2 - 1), 1)
        panel.SetSizer(sizer)

    def add_input_synapses(self):
        panel = wx.Panel(self.main_panel)
        panel.SetBackgroundColour('#FFDDFF')
        self.main_sizer.Add(panel, 1, wx.EXPAND | wx.LEFT)

    def add_feedback_synapses(self):
        pass

    def add_output_neurons(self):
        pass

    def add_internal_neurons_and_synapses(self):
        panel = wx.Panel(self.main_panel)
        panel.SetBackgroundColour('#DDFFDD')
        self.main_sizer.Add(panel, 8, wx.EXPAND | wx.RIGHT)

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
                        weight = self.esn.get_weight(x1, y1, x2, y2)
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
                            
                    arrow = SynapseGroup(panel, positions)
                    for direction in real_directions:
                        self.arrows[direction] = arrow
                        
                    sizer.Add(arrow, flag=wx.EXPAND)

        panel.SetSizer(sizer)

    def set_weights(self):
        for ((x1, y1), (x2, y2)), arrow in self.arrows.iteritems():
            weight = self.esn.get_weight(x1, y1, x2, y2)
            arrow.weights[((x1, y1), (x2, y2))] = weight

class SynapseGroup(wx.Panel):

    def __init__(self, parent, positions):
        super(SynapseGroup, self).__init__(parent)
        self.weights = {d: 0 for d in positions.keys()}
        self.positions = positions

        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def on_size(self, event):
        event.Skip()
        self.Refresh()

    def on_paint(self, event):
        w, h = self.GetClientSize()
        dc = wx.AutoBufferedPaintDC(self)
        dc.Clear()

        end_width = (w + h) / 10
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

class Neuron(wx.Panel):

    def __init__(self, parent):
        super(Neuron, self).__init__(parent)

        self.history = []

        self.SetWindowStyle(wx.SIMPLE_BORDER)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def on_size(self, event):
        event.Skip()
        self.Refresh()

    def on_paint(self, event):
        w, h = self.GetClientSize()
        dc = wx.AutoBufferedPaintDC(self)
        dc.Clear()
        dc.DrawRectangle(0, 0, w, h)

from esn import NeighbourESN
esn = NeighbourESN(
    n_input_units=1,
    width=5,
    height=5,
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

app = wx.App()
visualiser = Visualiser(esn)
visualiser.Show()
app.MainLoop()
