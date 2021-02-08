import pandas as pd 
import numpy as np 
# import lda 
from sklearn.feature_extraction.text import CountVectorizer 
import re
import nltk
from nltk.corpus import stopwords,wordnet
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import MyNLTK
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from sklearn.cluster import KMeans
import pickle
model=Word2Vec.load('5.model')
myNLTK=MyNLTK.MyNLTK()

fieldOne="Asian giant hornet workers can grow to 1.5 inches in length and are \
    similar in size to other wasps that occur in Pennsylvania and may be confused with \
    Asian giant hornets. Asian giant hornets are strikingly colored, with yellow heads, a \
    black thorax, and yellow and black or brown striped abdomens."

# temp=pd.read_csv("temp.csv")
# temp=temp["sen"].tolist()
# temp.append(fieldOne)
temp=[fieldOne]

a_vecs=[]
a_norms=[]
# important=()
for fieldOne in temp:
    fieldWords=myNLTK.process(fieldOne)
    fieldWords=list((list(zip(*fieldWords)))[0])

    vec=model.wv[fieldWords[0]]
    print(vec.shape)
    # print(vec.)
    for j in fieldWords[1:]:
        vec=vec+model.wv[j]
    vec=vec/len(fieldWords)
    a_vecs.append(vec)
    vec=np.linalg.norm(vec)
    a_norms.append(vec)



# data=pd.read_excel(r"E:\MCM2021\2021MCMProblemC_DataSet.xlsx")
# data=data.dropna(subset=["Notes"])
# labstatus=data["Lab Status"].tolist()
# data=data["Notes"].tolist()


# allData=[]
# allData.append(fieldWords)
# allDataN=[-1]
# num=0
# raw=[fieldOne]
# status=["Official"]
# # print(allData)
# for i,s in zip(data,labstatus):
#     if(type(i)!=str):
#         continue
#     # i
#     result=myNLTK.process(i) 
#     if (result==[]):
#         continue
#     result=list((list(zip(*result)))[0])
#     allData.append(result)
#     allDataN.append(num)
#     raw.append(i)
#     status.append(s)
#     num+=1


# df=pd.DataFrame({"num":allDataN,"raw":raw,"processed":allData,"status":status})
# df.to_csv("Processed.csv")


data=pd.read_csv("processed.csv")





d2=pd.read_csv("p4.csv")
# data["processed"]=data["processed"].apply(lambda x:eval(x))
allSim=data["processed"].tolist()
# allRaw=data["sen"].tolist()
# x=[]
# for r in allRaw:
#     if "large" in r or "big" in r:
#         print(r)
#         x.append(r)

# d=pd.DataFrame({"sen":x})
# d.to_csv("temp.csv")


# allS=data["status"].tolist()
allVec=[]
sim=[]
for i in allSim:
    x=eval(i)
    # print(type(x),x)
    if (type(x)==str):
        vector=model.wv[x]
        continue
    # print(type(model.wv[x[0]]))
    vector=model.wv[x[0]]
    for j in x[1:]:
        vector=vector+model.wv[j]

    vector=vector/len(i)
    # print(vector)
    b_norm = np.linalg.norm(vector)

    similiarity=0
    for vec,a_norm in zip(a_vecs,a_norms):
        similiarity += np.dot(vec, vector.T)/(a_norm * b_norm)

    sim.append(similiarity/len(a_vecs))

d2["similarity"]=sim
# d2.to_csv("p5.csv",encoding="utf_8_sig")

    





    # print(vector)
# print(data.info())

# n_clusters=2
# cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(allVec)

# r1=[]
# t=[]
# s1=[]
# for i,raw,s in zip(allVec,allRaw,allS):
#     t.append(cluster.predict([i.tolist()]))
#     r1.append(raw)
#     s1.append(s)
#     # print(raw,t)
#     # break
# df=pd.DataFrame({"raw":r1,"cluster":t,"s":s1}).to_csv("p3.csv",encoding="utf_8_sig")
# with open('clf.pickle', 'wb') as f:
#     pickle.dump(cluster, f)