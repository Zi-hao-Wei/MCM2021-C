import pandas as pd  
import numpy as np 
import os
import sys,shutil
import json

oldPath="E:\\MCM2021\\Photos\\"
newPath="E:\\MCM2021\\Labeled\\"
data=pd.read_excel(r"E:\MCM2021\2021MCMProblemC_DataSet.xlsx")
image=pd.read_excel(r"E:\MCM2021\2021MCM_ProblemC_ Images_by_GlobalID.xlsx")

# data = data.to_dict(orient='records')
image = image.to_dict(orient='records')
# print(data)
# print(image)

dictT={}
for i in image:
    if i["GlobalID"] in dictT:
        dictT[i["GlobalID"]].append((i["FileName"],i["FileType"]))
    else:
        dictT[i["GlobalID"]]=[(i["FileName"],i["FileType"])]

def l1(x):
    global dictT
    if x in dictT:
        return dictT[x]
    else:
        return {}
data["Sources"]=data["GlobalID"].apply(lambda x: l1(x))
data.to_csv("Processed.csv")
# dictData={}
# for i in data:
    # dictData[i["GlobalID"]]=i["Lab Status"]



# dirs = os.listdir(oldPath)
# for one_dir in dirs:
#     print(one_dir,dictT[one_dir],dictData[dictT[one_dir]])
#     old=oldPath+one_dir
#     new=newPath+ dictData[dictT[one_dir]] +"\\"+one_dir
#     shutil.move(old,new)

