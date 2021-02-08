from pyecharts import options as opts
from pyecharts.charts import Geo,Map,Timeline,BMap,Line
from pyecharts.datasets import register_url
from pyecharts.globals import ChartType
from pyecharts.commons.utils import JsCode
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np 

x=[]
y=[]
y0=1
for i in range(10):
    print(y0)
    x.append(i)
    y.append(y0)
    y0 = (1.0-0.00918*(y0-115))*y0

for i in range(len(y)):
    y[i]=float(int(y[i]*100))/100

marking=[[x[0],y[0]],[x[-1],y[-1]]]
print(marking)

mark1=opts.MarkPointItem(type_="min")
mark2=opts.MarkPointItem(type_="max")

line=(
    Line(init_opts=opts.InitOpts(width="1050px",height="700px"))
    .add_xaxis(x)
    .add_yaxis("Population growth curve of Asian Hornet",y,is_smooth=True,is_symbol_show=False)
    # .add_yaxis("",1)
    # .add_yaxis("",115)
    .set_series_opts(
        linestyle_opts=opts.LineStyleOpts(curve=1,width=2),
        label_opts=opts.LabelOpts(is_show=False),
        markpoint_opts=opts.MarkPointOpts(data=[mark1,mark2],symbol="roundRect",symbol_size=20,label_opts=opts.LabelOpts(is_show=False))
    )
    .set_global_opts(
        legend_opts=opts.LegendOpts( textstyle_opts= opts.TextStyleOpts(font_size=20)),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=20),name_gap=15,name_textstyle_opts=opts.TextStyleOpts(font_size=17)),
        yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=20),name_gap=10,name_textstyle_opts=opts.TextStyleOpts(font_size=17))
    )
)   

line.render("Population growth curve of Asian Hornet.html")