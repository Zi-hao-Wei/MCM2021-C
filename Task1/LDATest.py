import pandas as pd 
import numpy as np 
import lda 
from sklearn.feature_extraction.text import CountVectorizer 
import re



# texts=["orange banana apple grape","banana apple apple","grape", 'orange apple'] 
# cv = CountVectorizer()
# cv_fit=cv.fit_transform(texts)
# print(cv.vocabulary_)
# print(cv_fit)
# print(cv_fit.toarray())



data=pd.read_excel(r"E:\MCM2021\2021MCMProblemC_DataSet.xlsx")
data=data.dropna(subset=["Notes"])
g=data.groupby("Lab Status")
for gx,df in g:
    if (gx=="Negative ID"):
        data=df["Notes"].tolist()
        # print(len(data))

allData=[]
for i in data:
    # print("-------------")
    # print(type(i))
    if(type(i)!=str):
        continue
    # break
    result = re.sub(r'[\.|,|\’|\"|/|?|!|\(|\)]', ' ', i).strip()
    if (result==""):
        continue
    # print(cvz)
    allData.append(result)
    # print(result)


print(len(allData))
# print(data)

n_topics=20
n_iter=10
# data=["cross fire finding you are sees"]
cvectorizer=CountVectorizer(min_df=3,stop_words='english')
cvz=cvectorizer.fit_transform(allData)
# print(cvectorizer.vocabulary_)
# print(cvz)
# print(cvz.toarray())


lda_model=lda.LDA(n_topics=n_topics,n_iter=n_iter)
X_topics=lda_model.fit(cvz)

print(X_topics.components_)
print(X_topics.components_.shape)


# t=X_topics.components_.transpose()
# print(t.shape)
# # # pca initializtion usually leads to better results
# # tsne_model = TSNE(n_components =2, verbose =1, random_state =0, angle =.99, init='pca')
# # # 20-D -> 2-D
# # tsne_lda = tsne_model .fit_transform(t)

# # print(tsne_lda)
# # print(tsne_lda.shape)

# _lda_keys = []
# # 
# print(t.shape)
# for i in range(0,t.shape[0]):
# #     # print(t[i])
#     # print(t[i].argmax())
#     _lda_keys.append(t[i].argmax())
# # # # # # 并获得每个主题的顶级单词：
# print(_lda_keys)
# _lda_keys = [] 
# for i in range(X_topics.shape[0]): 
#     _lda_keys += _topics[i].argmax(),

# # 并获得每个主题的顶级单词： 
# topic_summaries = [] 
# topic_word = lda_model.topic_word_ # all topic words 
# vocab = cvectorizer.get_feature_names() 
# for i, topic_dist in enumerate(topic_word): 
#     topic_words = np .array(vocab)[np .argsort(topic_dist)][: -(n_top_words + 1): -1] # get! 
# topic_summaries .append(' ' .join(topic_words))