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

data=pd.read_excel(r"E:\MCM2021\2021MCMProblemC_DataSet.xlsx")
data=data.dropna(subset=["Notes"])
data=data["Notes"].tolist()
# g=data.groupby("Lab Status")
# for gx,df in g:
    # if (gx=="Negative ID"):
        # data=df["Notes"].tolist()

myNLTK=MyNLTK.MyNLTK()

fieldOne="Asian giant hornet workers can grow to 1.5 inches in length and are \
    similar in size to other wasps that occur in Pennsylvania and may be confused with \
    Asian giant hornets. Asian giant hornets are strikingly colored, with yellow heads, a \
    black thorax, and yellow and black or brown striped abdomens."


important=()

fieldWords=myNLTK.process(fieldOne)
fieldWords=list((list(zip(*fieldWords)))[0])


allData=[]
# allData.append(" ".join(fieldWords))
allData.append(fieldWords)
allDataN=[-1]
num=0
# print(allData)
for i in data:
    if(type(i)!=str):
        continue
    # i=i.lower()
    result=myNLTK.process(i) 
    # print(result)
    if (result==[]):
        continue
    result=list((list(zip(*result)))[0])
    # print(result)
    # allData.append(" ".join(result))
    allData.append(result)
    allDataN.append(num)

    num+=1

model = Word2Vec(allData,size=40,min_count=1, iter=150, window=10)
model.save('5.model')
# model=Word2Vec.load('4.model')
print(model.wv.most_similar("yellow"))
print(model.wv.most_similar("hornet"))


