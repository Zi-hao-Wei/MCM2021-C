from pyecharts import options as opts
from pyecharts.charts import Geo,Map,Timeline,BMap,Line
from pyecharts.datasets import register_url
from pyecharts.globals import ChartType
from pyecharts.commons.utils import JsCode
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np 
try:
    register_url("https://echarts-maps.github.io/echarts-countries-js/")
except Exception:
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    register_url("https://echarts-maps.github.io/echarts-countries-js/")


    # .add_coordinate('Ohio',-82.884126,40.425196))



# image=pd.read_excel(r"E:\MCM2021\2021MCM_ProblemC_ Images_by_GlobalID.xlsx")
data=pd.read_excel(r"E:\MCM2021\2021MCMProblemC_DataSet.xlsx")
# image = data.to_dict(orient='records')
ak="wKUfa4Zk2QZfY767GfmXWMGjpzm8DoKQ"


YMean=data["Latitude"].mean()
XMean=data["Longitude"].mean()

# print(XMean,YMean)
timeS=0




def geodistance(x,y):
    lng1,lat1=x
    lng2,lat2=y
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance




def plot_one(data):
    
    g=data["GlobalID"].tolist()
    x=data["Latitude"].tolist()
    y=data["Longitude"].tolist()
    t=data["Lab Status"].tolist()

    d={}

    global timeS
    
    
    
    for i in zip(g,x,y,t):
        if i[3] in d:
            d[i[3]].append((i[0],i[2],i[1]))
        else:
            d[i[3]]=[(i[0],i[2],i[1])]
    global ak

    bMap=(
        BMap(is_ignore_nonexistent_coord=False)
        .add_schema(baidu_ak=ak,center=[XMean,YMean])
    )


    for i in d:
        for j in d[i]:
            # print(j[1],j[2])
            bMap.add_coordinate(j[0],j[1],j[2])
            #   
    for i in d:
        dTemp=list((list(zip(*d[i])))[0])
        for x in range(len(dTemp)):
            dTemp[x]=(dTemp[x],1)
        bMap.add(i,dTemp,type_="scatter")

    bMap.set_series_opts(label_opts = opts.LabelOpts(is_show = False))
    # bMap.render("T\\"+str(timeS)+".html")
    
    timeS+=1
    return bMap



 
# for g,df in group:
#     print(group)

# geo.render("test.html")
# print(data["Detection Date"])
# data=data.dropna(axis=0,subset=["Detection Date"])

period=pd.to_datetime(data["Detection Date"],format=r"%Y-%m-%d %H:%M:%S", errors='coerce')
data["Detection Date"]=period
# data["Detection Date"]
# print(period)


# g=data.groupby("Lab Status")

# for gx,df in g:
#     if gx=="Positive ID":
#         data=df
#         break

# data=data.sort_values(by=["Detection Date"])
# # print(data)
# group = data.groupby("Detection Date")
# tl=Timeline()
# tl.add_schema(play_interval=300)

data1=data[(data['Detection Date'] > '2019-01-01 00:00:00') & (data['Detection Date'] < '2019-12-31 23:59:59')]
data2=data[(data['Detection Date'] > '2020-01-01 00:00:00')]


group = data1.groupby(data['Detection Date'].apply(lambda x:x.month))


XLabel=[]
YLabel=[]

dictA={}
dictA["Negative ID"]=[]
dictA["Positive ID"]=[]
dictA["Unverified"]=[]
dictA["Unprocessed"]=[]
dictA["Sum"]=[]

# for g,df in group:

#     dictA["Sum"].append(len(df))
#     x=df.groupby("Lab Status")
    
#     dictB={}
#     dictB["Negative ID"]=0
#     dictB["Positive ID"]=0
#     dictB["Unverified"]=0
#     dictB["Unprocessed"]=0
    
#     for t,d in x:
#         dictB[t]=len(d)
#     for t in dictB:
#         dictA[t].append(dictB[t])

#     XLabel.append("2019-"+str(int(g)))
    # YLabel.append(dictA)    
    # print(g)
    # print(df)

group = data2.groupby(data['Detection Date'].apply(lambda x:x.month))

print(dictA)

for g,df in group:

    dictA["Sum"].append(len(df))
    x=df.groupby("Lab Status")
    
    dictB={}
    dictB["Negative ID"]=0
    dictB["Positive ID"]=0
    dictB["Unverified"]=0
    dictB["Unprocessed"]=0
    
    for t,d in x:
        dictB[t]=len(d)
    for t in dictB:
        dictA[t].append(dictB[t])

    XLabel.append("2020-"+str(int(g)))
    # YLabel.append(dictA)    
    # print(g)
    # print(df)
print(dictA)

# print(XLabel)
# print(YLabel)

c = (
    Line().add_xaxis(XLabel)
        .add_yaxis("Negative ID",dictA["Negative ID"])
        .add_yaxis("Positive ID",dictA["Positive ID"])
        .add_yaxis("Unverified",dictA["Unverified"])
        .add_yaxis("Unprocessed",dictA["Unprocessed"])
        # .add_yaxis("Sum",dictA["Sum"])
        .set_series_opts(
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2))
        .set_global_opts(
            legend_opts=opts.LegendOpts( textstyle_opts= opts.TextStyleOpts(font_size=20)),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=20),name_gap=15,name_textstyle_opts=opts.TextStyleOpts(font_size=17)),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=20),name_gap=10,name_textstyle_opts=opts.TextStyleOpts(font_size=17))
        )
        # .set_global_opts(
        # )
)

c.render('1.html')



# x=data1["Longitude"].tolist()
# y=data["Latitude"].tolist()
# t=data["Detection Date"].tolist()
# pre=(x[0],y[0])
# tpre=t[0]

# for X1,Y1,t1 in zip(x[1:],y[1:],t[1:]):
#     now=(X1,Y1)
#     print("Distance: ",tpre," ",t1 ," ",geodistance(now,pre))
#     pre=now
#     tpre=t1

# tl.add(plot_one(data1),"2019")
# tl.add(plot_one(data2),"2020")

# # for g,df in group:
#     # tl.add(plot_one(g,df),"%s" % g)


# tl.render("positive.html")