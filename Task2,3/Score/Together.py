import pandas as pd

original=pd.read_csv("Distance2.csv")
print(original.info())
original=original.drop(columns=["Unnamed: 0"])
graph=pd.read_csv("CVSimLabeled.csv")
mapping=pd.read_excel(r"E:\MCM2021\2021MCM_ProblemC_ Images_by_GlobalID.xlsx")


a=graph["name"]
b=graph["sim"]

maxb=b.max()
minb=b.min()
b=graph["sim"].apply(lambda x:(x-minb)/(maxb-minb))



dictA={}
for x,y in zip(a,b):
    dictA[x]=y

a=mapping["FileName"]
b=mapping["GlobalID"]
dictB={}
for x,y in zip(a,b):
    if x in dictA:
        l=dictA[x]
    else:
        l=0.5

    if y in dictB:
        dictB[y].append(l)
    else:
        dictB[y]=[l]

x=original["GlobalID"].tolist()    
y=[]
for i in x:
    if i in dictB:
        y.append(max(dictB[i]))
    else:
        y.append(0.5)


original["CVsim"]=y


# t=original["NLPScore"]
# maxb=t.max()
# minb=t.min()
# original["NLPScore"]=t.apply(lambda x:(x-minb)/(maxb-minb))



original.to_csv("D3.csv")




# s=[]

# raw=original["Notes"].tolist()
# for i in raw:
#     if type(i)==str:
#         if i in dictA:
#             s.append(dictA[i])
#         else:
#             s.append(0)
#     else:
#         s.append(0)
# original["NLPScore"]=s
# original.to_csv("NLPProcessed.csv")


# # for x,y in zip(a,b):
    