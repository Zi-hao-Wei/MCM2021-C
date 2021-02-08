 
import logging
import time
import os
import jieba
import glob
import random
import copy
import chardet
import gensim
from gensim import corpora,similarities, models
from pprint import pprint
import jieba.analyse
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.decomposition import PCA
 
 
 
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
start = time.clock()
 
# print ('#----------------------------------------#'
# print '#                                        #'
# print '#              载入语料库                #'
# print '#                                        #'
# print '#----------------------------------------#\n'
def PreprocessDoc(root):
 
    allDirPath = [] # 存放语料库数据集文件夹下面左右的文件夹路径,string,[1:]为所需
    fileNumList = []
 
    def processDirectory(args, dirname, filenames, fileNum=0):
        allDirPath.append(dirname)
        for filename in filenames:
            fileNum += 1
        fileNumList.append(fileNum)
    os.path.walk(root, processDirectory, None)
    totalFileNum = sum(fileNumList)
    print ('总文件数为: ' + str(totalFileNum))
 
    return allDirPath
 
 
# print '#----------------------------------------#'
# print '#                                        #'
# print '#              合成语料文档                #'
# print '#                                        #'
# print '#----------------------------------------#\n'
 
# 每个文档一行,第一个词是这个文档的类别
 
def SaveDoc(allDirPath, docPath, stopWords):
 
    print '开始合成语料文档:'
 
    category = 1 # 文档的类别
    f = open(docPath,'w') # 把所有的文本都集合在这个文档里
 
    for dirParh in allDirPath[1:]:
 
        for filePath in glob.glob(dirParh + '/*.txt'):
 
            data = open(filePath, 'r').read()
            texts = DeleteStopWords(data, stopWords)
            line = '' # 把这些词缩成一行,第一个位置是文档类别,用空格分开
            for word in texts:
                if word.encode('utf-8') == '\n' or word.encode('utf-8') == 'nbsp' or word.encode('utf-8') == '\r\n':
                    continue
                line += word.encode('utf-8')
                line += ' '
            f.write(line + '\n') # 把这行写进文件
        category += 1 # 扫完一个文件夹,类别+1
 
    return 0 # 生成文档,不用返回值
 
 
# print '#----------------------------------------#'
# print '#                                        #'
# print '#             分词+去停用词               #'
# print '#                                        #'
# print '#----------------------------------------#\n'
def DeleteStopWords(data, stopWords):
 
    wordList = []
 
    # 先分一下词
    cutWords = jieba.cut(data)
    for item in cutWords:
        if item.encode('utf-8') not in stopWords: # 分词编码要和停用词编码一致
            wordList.append(item)
 
    return wordList
 
 
# print '#----------------------------------------#'
# print '#                                        #'
# print '#                 tf-idf                 #'
# print '#                                        #'
# print '#----------------------------------------#\n'
def TFIDF(docPath):
 
    print '开始tfidf:'
 
    corpus = [] # 文档语料
 
    # 读取语料,一行语料为一个文档
    lines = open(docPath,'r').readlines()
    for line in lines:
        corpus.append(line.strip()) # strip()前后空格都没了,但是中间空格还保留
 
    # 将文本中的词语转换成词频矩阵,矩阵元素 a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()
 
    # 该类会统计每个词语tfidf权值
    transformer = TfidfTransformer()
 
    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
 
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()
 
    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()
    print weight
 
    # # 输出所有词
    # result = open(docPath, 'w')
    # for j in range(len(word)):
    #     result.write(word[j].encode('utf-8') + ' ')
    # result.write('\r\n\r\n')
    #
    # # 输出所有权重
    # for i in range(len(weight)):
    #     for j in range(len(word)):
    #         result.write(str(weight[i][j]) + ' ')
    #     result.write('\r\n\r\n')
    #
    # result.close()
 
    return weight
 
 
# print '#----------------------------------------#'
# print '#                                        #'
# print '#                   PCA                  #'
# print '#                                        #'
# print '#----------------------------------------#\n'
def PCA(weight, dimension):
 
    from sklearn.decomposition import PCA
 
    print '原有维度: ', len(weight[0])
    print '开始降维:'
 
    pca = PCA(n_components=dimension) # 初始化PCA
    X = pca.fit_transform(weight) # 返回降维后的数据
    print '降维后维度: ', len(X[0])
    print X
 
    return X
 
 
# print '#----------------------------------------#'
# print '#                                        #'
# print '#                 k-means                #'
# print '#                                        #'
# print '#----------------------------------------#\n'
def kmeans(X, k): # X=weight
 
    from sklearn.cluster import KMeans
 
    print '开始聚类:'
 
    clusterer = KMeans(n_clusters=k, init='k-means++') # 设置聚类模型
 
    # X = clusterer.fit(weight) # 根据文本向量fit
    # print X
    # print clf.cluster_centers_
 
    # 每个样本所属的簇
    y = clusterer.fit_predict(X) # 把weight矩阵扔进去fit一下,输出label
    print y
 
    # i = 1
    # while i <= len(y):
    #     i += 1
 
    # 用来评估簇的个数是否合适,距离约小说明簇分得越好,选取临界点的簇的个数
    # print clf.inertia_
 
    return y
 
 
# print '#----------------------------------------#'
# print '#                                        #'
# print '#                 BIRCH                 #'
# print '#                                        #'
# print '#----------------------------------------#\n'
def birch(X, k): # 待聚类点阵,聚类个数
 
    from sklearn.cluster import Birch
 
    print '开始聚类:'
 
    clusterer = Birch(n_clusters=k)
 
    y = clusterer.fit_predict(X)
    print '输出聚类结果:'
    print y
 
    return y
 
 
# print '#----------------------------------------#'
# print '#                                        #'
# print '#                轮廓系数                 #'
# print '#                                        #'
# print '#----------------------------------------#\n'
def Silhouette(X, y):
 
    from sklearn.metrics import silhouette_samples, silhouette_score
 
    print '计算轮廓系数:'
 
    silhouette_avg = silhouette_score(X, y) # 平均轮廓系数
    sample_silhouette_values = silhouette_samples(X, y) # 每个点的轮廓系数
 
    print(silhouette_avg)
 
    return silhouette_avg, sample_silhouette_values
 
 
# print '#----------------------------------------#'
# print '#                                        #'
# print '#                  画图                  #'
# print '#                                        #'
# print '#----------------------------------------#\n'
def Draw(silhouette_avg, sample_silhouette_values, y, k):
 
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
 
    # 创建一个 subplot with 1-row 2-column
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(18, 7)
 
    # 第一个 subplot 放轮廓系数点
    # 范围是[-1, 1]
    ax1.set_xlim([-0.2, 0.5])
 
    # 后面的 (k + 1) * 10 是为了能更明确的展现这些点
    ax1.set_ylim([0, len(X) + (k + 1) * 10])
 
    y_lower = 10
 
    for i in range(k): # 分别遍历这几个聚类
 
        ith_cluster_silhouette_values = sample_silhouette_values[y == i]
        ith_cluster_silhouette_values.sort()
 
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
 
        color = cm.spectral(float(i)/k) # 搞一款颜色
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0,
                          ith_cluster_silhouette_values,
                          facecolor=color,
                          edgecolor=color,
                          alpha=0.7) # 这个系数不知道干什么的
 
        # 在轮廓系数点这里加上聚类的类别号
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
 
        # 计算下一个点的 y_lower y轴位置
        y_lower = y_upper + 10
 
    # 在图里搞一条垂直的评论轮廓系数虚线
    ax1.axvline(x=silhouette_avg, color='red', linestyle="--")
 
    plt.show()
 
 
 
 
 
 
if __name__ == "__main__":
 
    root = '/Users/John/Desktop/test'
    stopWords = open('/Users/John/Documents/NLPStudy/stopwords-utf8', 'r').read()
    docPath = '/Users/John/Desktop/test/doc.txt'
    k = 3
 
    allDirPath = PreprocessDoc(root)
    SaveDoc(allDirPath, docPath, stopWords)
 
    weight = TFIDF(docPath)
    X = PCA(weight, dimension=800) # 将原始权重数据降维
    # y = kmeans(X, k) # y=聚类后的类标签
    y = birch(X, k)
    silhouette_avg, sample_silhouette_values = Silhouette(X, y) # 轮廓系数
    Draw(silhouette_avg, sample_silhouette_values, y, k)
 
 
end = time.clock()
print '运行时间: ' + str(end - start)
 
 
 