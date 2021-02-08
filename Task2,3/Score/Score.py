import pandas as pd 
import numpy as np 
from pyecharts import options as opts
from pyecharts.charts import Geo,Map,Timeline,BMap,Line,Scatter
from pyecharts.datasets import register_url
from pyecharts.globals import ChartType
from pyecharts.commons.utils import JsCode
import seaborn as sns 
import matplotlib.pyplot as plt 
data=pd.read_excel("T1.xlsx")

# data["Score"]=0.2437*data["NLPScore"]+0.4379*data["CVsim"]+0.2190*(1-data["distance"])+0.0994*(1-data["time"])
# data.to_excel("S2.xlsx")
d = data[["NLPScore","CVsim","distance","time"]]
d = d.dropna()
# r = data[""]

from sklearn.manifold import TSNE
tsne=TSNE()

tsne=TSNE()
tsne.fit_transform(d)  #进行数据降维,降成两维


d["time"]=1-d["time"]
d["distance"]=1-d["distance"]

#a=tsne.fit_transform(data_zs) #a是一个array,a相当于下面的tsne_embedding_
tsne=pd.DataFrame(tsne.embedding_,index=d.index) #转换数据格式
 
import matplotlib.pyplot as plt 
 
negative=tsne[data[u'Lab Status']=="Negative ID"]
plt.plot(negative[0],negative[1],'r.')
 
unverified=tsne[data[r'Lab Status']=="Unverified"]
plt.plot(unverified[0],unverified[1],'go')

positive=tsne[data[r'Lab Status']=="Positive ID"]
plt.plot(positive[0],positive[1],'b*')
plt.xlabel("t-SNE 0")
plt.ylabel("t-SNE 1")
plt.legend(labels=["Negative ID","Unverified","Positive ID"])

# print(negative)
negative["Lab Status"]="Negative"
unverified["Lab Status"]="Unverified"
positive["Lab Status"]="Positive"

allD=pd.concat([negative,unverified,positive])
allD.to_csv("TSNE1.csv")
# print(allD)
# # tips=sns.load_dataset("tips")
# # print(tips)
# allD=allD.rename(columns={0:'t-SNE 0',1:'t-SNE 1'})
# sns.relplot(x='t-SNE 0',y='t-SNE 1',data=allD,hue="Lab Status")
plt.show()
