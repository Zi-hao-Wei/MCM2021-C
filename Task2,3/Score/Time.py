import pandas as pd

data=pd.read_csv(r"D3.csv")
period=pd.to_datetime(data["Detection Date"],format=r"%Y-%m-%d %H:%M:%S", errors='coerce')
data["Detection Date"]=period

def getPositive():
    global data 
    g=data.groupby("Lab Status")
    for i,df in g:
        if i=="Positive ID":
            return df 

pos=getPositive()
date=pos["Detection Date"].tolist()
alldate=data["Detection Date"].tolist()
time=[]
for i in alldate:
    l=[]
    for j in date:
        if i>j:
            l.append(i-j)
        else:
            l.append(j-i)
    time.append(min(l).days)

maxt=max(time)
mint=min(time)

data["time"]=time
data["time"]=data["time"].apply(lambda x:(x-mint)/(maxt-mint))

data.to_excel("T1.xlsx")