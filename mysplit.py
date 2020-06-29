from bs4 import BeautifulSoup
import requests
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing import sequence
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import numpy as np
from pandas import DataFrame
import pandas as pd
import csv
from scipy.spatial.distance import cdist

fp = open('input.csv','r')
reader = csv.reader(fp)
lst = [i[0] for i in reader]

# 构造特殊句子锁定最佳cluster part1
lst.append('fuck in the hell~')

# 分句
# X_Org = sent_tokenize(lst,'english')
# print(X_Org)

# 矢量化
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(lst)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)
weight = tfidf.toarray()

# 聚类
n_target = int(len(weight)/5)
for n in range(n_target-3, n_target+3):
    kmeans = KMeans(n_clusters=n).fit(weight)
    labels = kmeans.labels_
    result = DataFrame({'category':labels,'sentence':lst})
    pd.set_option('display.max_rows', None)

    # 构造特殊句子锁定最佳cluster part2
    a1 = result.query('sentence== "fuck in the hell~"')['category'].iat[0]
    if len(result[result.category == a1]) < 2:
        print(n)
        print(result.sort_values(by='category'))
        break

# groups方式输出
# grouped = result.groupby('category')
# for name,group in grouped:
#     print (name)
#     print (group.iloc[0])


