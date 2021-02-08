import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import folium 
import time 
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt

pos19=pd.read_csv("pos19.csv")

# df=pd.read_csv("all20.csv")

# g=df.groupby("")


pos20=pd.read_csv("pos20.csv")
un20=pd.read_csv("all20.csv")



data=pd.concat([pos20,un20,pos19])
x=data["Longitude"].tolist()
y=data["Latitude"].tolist()

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

# X=list(zip(x,y))
# from sklearn.cluster import KMeans
# n_clusters=4
# cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(X)
# centroid=cluster.cluster_centers_
# inertia=cluster.inertia_
# color=['red','pink','orange','gray']
# fig, axi1=plt.subplots(1)
# # for i in range(n_clusters):
# # axi1.scatter(x, y, marker='o',s=8, c='r')
# for i in X:
#     t=cluster.predict([i])
#     print(type(t))
#     axi1.scatter(i[0],i[1],c=color[t[0]])

# axi1.scatter(centroid[:,0],centroid[:,1],marker='x',s=100,c='black')

# plt.show()

# p2=list(zip(x,y))

# while len(points)>0:
#     best=0
#     for i in left:
#         for j in left:
#             if geodistance(i,j)


# print(data)





mfLat=pos19["Latitude"].mean()
mfLon=pos19["Longitude"].mean()
map_osm = folium.Map(location=[mfLat, mfLon], zoom_start=10,control_scale=True, tiles='Stamen Terrain')

def change(data,c):
    global map_osm
    coordinates = list(zip(data['Latitude'], data['Longitude']))
    data['coordinates'] = coordinates
    incidents = folium.map.FeatureGroup()

    l=data["coordinates"].tolist()
    print(l)
    # Loop through the 200 crimes and add each to the incidents feature group
    for lat, lng, in l:
        incidents.add_child(
            folium.CircleMarker(
                [lat, lng],
                radius=10, # define how big you want the circle markers to be
                color=c,
                fill=True,
                fill_color=c,
                fill_opacity=0.4
            )
        )
    
    map_osm.add_child(incidents)
    return data


pos19=pos19.drop([3])

print(pos19)
pos19=change(pos19,"red")

pos20=pos20.drop([1])
print(pos20)

pos20=change(pos20,"red")
# un20=change(un20,"blue")





map_osm.save("map3.html")