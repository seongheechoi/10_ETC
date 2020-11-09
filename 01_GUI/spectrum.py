import pandas as pd
from pandas import DataFrame
import sys
import seaborn as sns
import itertools
import numpy as np
import matplotlib.pyplot as plt


# 데이터 정리
#### ---------------------------------------------------------------------------------####

fft_data = pd.read_csv('u24a.csv', header=None, encoding='cp949', low_memory=False)
fft_data1 = fft_data.dropna(axis=1,how='all')            #NA열 지우기
fft_data2 = fft_data1.dropna(axis=0,how='all')               #NA행 지우기
fft_d3 = fft_data2.drop(fft_data2.index[0:95])
fft_d4 = fft_d3.drop(fft_data2.index[96:98])
fft_d5 = fft_d4.drop(fft_data2.index[99:113])                 #행 구간 지우기
fft_d5.columns = fft_d5.iloc[0,:]+fft_d5.iloc[1,:]           # 첫째행을 헤더로 하기
fft_d6=fft_d5.T.drop_duplicates().T
fft_d7 = fft_d6.drop(fft_d6.index[0:2])
col_names = fft_d7.columns.tolist()
col_names[0] = "Freq"
fft_d7.columns = col_names
fft_d7 = fft_d7.set_index('Freq')
fft_d7 = fft_d7.astype(float)

#### ---------------------------------------------------------------------------------####

print(fft_d7)

# 그래프
#### ---------------------------------------------------------------------------------####
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.palettes import Dark2_5 as palette
from bokeh.palettes import Viridis4 as palette2
from bokeh.layouts import widgetbox, row, column
from bokeh.models import CheckboxButtonGroup, CustomJS
from bokeh.models.widgets import CheckboxGroup
from bokeh.models.annotations import Title, Legend
import itertools
from bokeh.models import CheckboxGroup, CustomJS

# output_notebook()

p = figure(plot_width=1600, plot_height=900, title="Noise spectrums")

colors = itertools.cycle(palette2)
nseries = len(fft_d7.columns)

series = []

# add a line renderer
for n in range(nseries):
    y = fft_d7.columns[n]
    series.append(p.line(x=fft_d7.index, y=fft_d7[fft_d7.columns[n]], line_width=0.5, legend=str(fft_d7.columns[n]),
                         color=next(colors), name=str(fft_d7.columns[n])))

p.yaxis.axis_label = "dB(A)"
p.xaxis.axis_label = "Frequency(Hz)"

p.legend.location = "top_right"
# p.legend.click_policy="hide"

js = ""
for n in range(nseries):
    js_ = """
        if (checkbox.active.indexOf({n}) >-1) {{
            l{n}.visible = true
        }} else {{
            l{n}.visible = false
        }} """
    js += js_.format(n=n)

callback = CustomJS(code=js, args={})
checkbox_group = CheckboxGroup(labels=[str(fft_d7.columns[n]) for n in range(nseries)], active=[0, 1, 2, 3],
                               callback=callback)
callback.args = dict([('l{}'.format(n), series[n]) for n in range(nseries)])
callback.args['checkbox'] = checkbox_group

show(row([p, checkbox_group]))  # show the results

### ------------------------------------------------------------------------------------------###