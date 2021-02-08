from pyecharts import options as opts
from pyecharts.charts import Geo,Map,Timeline,BMap
from pyecharts.datasets import register_url
from pyecharts.globals import ChartType
from pyecharts.commons.utils import JsCode
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
try:
    register_url("https://echarts-maps.github.io/echarts-countries-js/")
except Exception:
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    register_url("https://echarts-maps.github.io/echarts-countries-js/")


data=pd.read_excel(r"E:\MCM2021\2021MCMProblemC_DataSet.xlsx")
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


bMap=(
        BMap(is_ignore_nonexistent_coord=False)
        .add_schema(baidu_ak=ak,center=[XMean,YMean])
    )

def plot_one(data):
    
    global bMap

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




    for i in d:
        for j in d[i]:
            bMap.add_coordinate(j[0],j[1],j[2])
    for i in d:
        dTemp=list((list(zip(*d[i])))[0])
        for x in range(len(dTemp)):
            dTemp[x]=(dTemp[x],1)
        bMap.add(i,dTemp,type_="scatter")

    bMap.set_series_opts(label_opts = opts.LabelOpts(is_show = False))
    timeS+=1
    return bMap



period=pd.to_datetime(data["Detection Date"],format=r"%Y-%m-%d %H:%M:%S", errors='coerce')
data["Detection Date"]=period


g=data.groupby("Lab Status")

unverified=[]
positive=[]

for gx,df in g:
    if gx=="Positive ID":
        positive=df
        # break
    if gx=="Unverified":
        unverified=df

positive=positive.sort_values(by=["Detection Date"])
unverified=unverified.sort_values(by=["Detection Date"])


tl=Timeline()
tl.add_schema(play_interval=300)

positive2019=positive[(positive['Detection Date'] > '2019-01-01 00:00:00') & (positive['Detection Date'] < '2019-12-31 23:59:59')]
unverified2020=unverified[(unverified['Detection Date'] > '2020-01-01 00:00:00')]
positive2020=positive[(positive['Detection Date'] > '2020-01-01 00:00:00')]

positive2019=positive2019.drop([3])
print(positive2019)

X1Mean=positive2019["Longitude"].mean()
Y1Mean=positive2019["Latitude"].mean()

print(X1Mean,Y1Mean)

all2020=unverified2020

unverified2020["T"]=0
positive2020["T"]=1

# all2020=pd.concat([unverified2020,positive2020])
all2020= unverified2020

all2020["d"]=all2020[["Longitude","Latitude"]].apply(lambda x:geodistance(x,(X1Mean,Y1Mean)),axis=1)

# print(all2020)


# minX=all2020["d"].min()
# maxX=all2020["d"].max()
# all2020["d"]=all2020["d"].apply(lambda x:(x-minX)/(maxX-minX))

# x1=all2020["Longitude"].tolist()
# y1=all2020["Latitude"].tolist()



# temp=list(zip(x1,y1))
# distanceX=np.zeros((len(temp),len(temp)))
# num=np.zeros(len(temp))
# for i in range(len(temp)):
#     for j in range(len(temp)):
#         distanceX[i][j]=geodistance(temp[i],temp[j])
#         num[i]=np.sum(distanceX[i]<=1)-1
# print(distanceX)

# all2020["k"]=num
# minX=all2020["k"].min()
# maxX=all2020["k"].max()
# all2020["k"]=all2020["k"].apply(lambda x:(x-minX)/(maxX-minX))
# all2020.to_csv("d4.csv")



all20=pd.read_csv("d3.csv")

all20["score"] = (1-all20["d"])*np.power(2,-all20["k"])

# all20["score"]= (1/np.log(all20["d"]))*(np.exp(-(all20["k"]-0.5)*(all20["k"]-0.5)/2))
# all20["score"]=(1-np.power(all20["d"],2.718))*np.log(1+all20["k"])
all20=all20.sort_values(by=["score"])
all20.to_csv("alltest.csv")
all20=all20.tail(14)
print(all20.tail(14))


all20.to_csv("all20.csv")
tl.add(plot_one(positive2019),"2019")
tl.add(plot_one(all20),"2020_p")
tl.render("positive2.html")