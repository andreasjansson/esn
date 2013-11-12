import wx
import math

class Visualiser(wx.Frame):

    def __init__(self, neighbour_esn):
        super(Visualiser, self).__init__(None, -1, 'esn')
#        self.SetDoubleBuffered(True)

        self.esn = neighbour_esn
        self.neurons = {}
        self.arrows = {}
        self.add_components()
        self.set_weights()

    def add_components(self):

        panel = wx.Panel(self)
        layout = wx.BoxSizer(wx.HORIZONTAL)

        settings_panel = wx.Panel(panel)
        settings_panel.SetBackgroundColour('#FFDDDD')
        layout.Add(settings_panel, 1, wx.EXPAND | wx.LEFT)

        main_panel = wx.Panel(panel)
        main_panel.SetBackgroundColour('#DDFFDD')
        layout.Add(main_panel, 5, wx.EXPAND | wx.RIGHT)

        cols = self.esn.width * 2 - 1
        rows = self.esn.height * 2 - 1
        main_sizer = wx.GridSizer(cols, rows)

        for row in range(rows):
            if row % 2 == 0:
                y = row / 2

            for col in range(cols):
                if col % 2 == 0:
                    x = col / 2

                if col % 2 == 0 and row % 2 == 0:
                    neuron = Neuron(main_panel)
                    self.neurons[(x, y)] = neuron
                    main_sizer.Add(neuron, flag=wx.EXPAND)

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
                    arrow = Arrow(main_panel, real_directions)
                    for direction in real_directions:
                        self.arrows[direction] = arrow
                        
                    main_sizer.Add(arrow, flag=wx.EXPAND)

        main_panel.SetSizer(main_sizer)

        panel.SetSizer(layout)

    def set_weights(self):
        for ((x1, y1), (x2, y2)), arrow in self.arrows.iteritems():
            weight = self.esn.get_weight(x1, y1, x2, y2)
            arrow.weights[((x1, y1), (x2, y2))] = weight

class Arrow(wx.Panel):

    def __init__(self, parent, directions):
        super(Arrow, self).__init__(parent)
        directions = directions
        self.weights = {d: 0 for d in directions}

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
        ex1 = end_width / 2
        ey = end_height
        ex2 = -end_width / 2
        a = math.atan2(w, h)
        rotx1 = ex1 * math.cos(a) + ey * math.sin(a)
        roty1 = -ex1 * math.sin(a) + ey * math.cos(a)
        rotx2 = ex2 * math.cos(a) + ey * math.sin(a)
        roty2 = -ex2 * math.sin(a) + ey * math.cos(a)

        for ((x1, y1), (x2, y2)), weight in self.weights.iteritems():
            if weight > 0:
                colour = wx.Colour(255, 255 - int(min(weight, 1) * 255.0), 255)
            else:
                colour = wx.Colour(255 - int(min(-weight, 1) * 255.0), 255, 255)

            pen = wx.Pen(colour)
#            if abs(weight) > 1:
#                print weight
#                pen.SetWidth(2)
#                pen.SetStyle(wx.SHORT_DASH)
            dc.SetPen(pen)

            if x1 == x2:
                dc.DrawLine(w / 2, 0, w / 2, h)
                if y1 > y2:
                    dc.DrawPolygon([(w / 2, 0),
                                    (w / 2 - end_width / 2, end_height),
                                    (w / 2 + end_width / 2, end_height)])
                else:
                    dc.DrawPolygon([(w / 2, h),
                                    (w / 2 - end_width / 2, h - end_height),
                                    (w / 2 + end_width / 2, h - end_height)])

            elif y1 == y2:
                dc.DrawLine(0, h / 2, w, h / 2)
                if x1 > x2:
                    dc.DrawPolygon([(0, h / 2),
                                    (end_height, h / 2 - end_width / 2),
                                    (end_height, h / 2 + end_width / 2)])
                else:
                    dc.DrawPolygon([(w, h / 2),
                                    (w - end_height, h / 2 - end_width / 2),
                                    (w - end_height, h / 2 + end_width / 2)])
            elif (x1 < x2 and y1 < y2) or (x2 < x1 and y2 < y1):
                dc.DrawLine(0, 0, w, h)
                if x1 > x2:
                    dc.DrawPolygon([(0, 0),
                                    (rotx1, roty1),
                                    (rotx2, roty2)])
                else:
                    dc.DrawPolygon([(w, h),
                                    (w - rotx1, h - roty1),
                                    (w - rotx2, h - roty2)])
            else:
                dc.DrawLine(0, h, w, 0)
                if x1 > x2:
                    dc.DrawPolygon([(0, h),
                                    (rotx1, h - roty1),
                                    (rotx2, h - roty2)])
                else:
                    dc.DrawPolygon([(w, 0),
                                    (w - rotx1, roty1),
                                    (w - rotx2, roty2)])

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
