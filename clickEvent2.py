import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
#np.random.seed(1)
x1 = np.random.rand(100)
y1 = np.random.rand(100)
x2 = np.random.rand(100)
y2 = np.random.rand(100)
x3 = np.random.rand(100)
y3 = np.random.rand(100)

#f = go.FigureWidget([go.Scatter(x=x, y=y, mode='markers')])
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=x1, y=y1, mode='markers', name='First', line=dict(width=1)), secondary_y=False)
fig.add_trace(go.Scatter(x=x2, y=y2, mode='markers', name='Second', line=dict(width=1)), secondary_y=False)
fig.add_trace(go.Scatter(x=x3, y=y3, mode='markers', name='Third', line=dict(width=1)), secondary_y=False)
fig.update_layout(title_text="Click event test")

scatter = fig.data[0:2]
print(scatter)
#
colors = ['#a3a7e4'] * 100
scatter[0].marker.color = colors
scatter[0].marker.size = [10] * 100
fig.layout.hovermode = 'closest'
#
#
# # create our callback function
def update_point(trace, points, selector):
     c = list(scatter[0].marker.color)
     s = list(scatter[0].marker.size)
     for i in points.point_inds:
         c[i] = '#bae2be'
         s[i] = 20
         with f.batch_update():
             scatter[0].marker.color = c
             scatter[0].marker.size = s

scatter[0].on_click(update_point)

fig.show()
