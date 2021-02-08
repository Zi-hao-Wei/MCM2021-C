import pandas as pd
import numpy as np
import pandas as pd
from textblob import TextBlob


sample=["yellow","strip","black","giant","huge","big","large","orange","yellowish"]
official=["WASD","specimen"] 
positive=["sure","confident","certainly","definitely","indeed","absolutely","without a doubt"]
negative=["not sure","Not sure"]

data=pd.read_csv("p5.csv")
raw=data["raw"].tolist()
pa=[]
for r in raw:
    p=0
    for x in sample:
        if x in r:
            p=p+0.5
    for x in official:
        if x in r:
            p=p+2
    for x in positive:
        if x in r:
            p=p+0.2
    for x in negative:
        if x in r:
            p=p-0.5
    pa.append(p)

maxp=max(pa)
minp=min(pa)

pa=list(map(lambda x:(x-minp)/(maxp-minp),pa))

data["p"]=pa


def temp(x):
    if (x=="[0]"):
        return 0
    else:
        return 1

def temp2(x):
    if (x>0):
        return 0
    elif x<-0.4:
        return -0.4
    else:
        return x/5


maxp=data["similarity"].max()
minp=data["similarity"].min()

data["similarity"]=data["similarity"].apply(lambda x:(x-minp)/(maxp-minp))
data["cluster"]=data["cluster"].apply(lambda x: temp(x))
data["pol"]=data["pol"].apply(lambda x:temp2(x))
data["score"]=data["similarity"]*data["p"]+data["pol"]/5
data.to_csv("score.csv")