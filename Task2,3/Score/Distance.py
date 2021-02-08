import pandas as pd 
import numpy as np 
from math import radians, cos, sin, asin, sqrt
import pandas

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


data=pd.read_csv("CVSim3.csv")

def getPositive():
    global data 
    g=data.groupby("Lab Status")
    for i,df in g:
        if i=="Positive ID":
            return df 

allPos=[]
pos=getPositive()
lng=pos["Longitude"].tolist()
lat=pos["Latitude"].tolist()
for i in zip(lng,lat):
    allPos.append(i)

lng=data["Longitude"].tolist()
lat=data["Latitude"].tolist()
t=[]
for i in zip(lng,lat):
    ans=100000
    for j in allPos:
        ans=min(ans,geodistance(i,j))
    t.append(ans)

maxt=max(t)
mint=min(t)

data["distance"]=t 
data["distance"]=data["distance"].apply(lambda x:(x-mint)/(maxt-mint))
data.to_csv("Distance2.csv")