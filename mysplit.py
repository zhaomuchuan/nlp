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

# 构造特殊句子锁定最佳cluster part1
df = pd.read_csv('input.csv')
df = df.append([{'Time':'','Host':'','Problem':'fuck in the hell~'}], ignore_index=True)
lst = df['Problem'].tolist()

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
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # copy与=区别
    result = df.copy()
    result.insert(0, 'category', labels)

    # 构造特殊句子锁定最佳cluster part2
    a1 = result.query('Problem== "fuck in the hell~"')['category'].iat[0]
    if len(result[result.category == a1]) < 2:
        print(n)
        print(result.sort_values(by='category'))
        break
#
# # groups方式输出
# # grouped = result.groupby('category')
# # for name,group in grouped:
# #     print (name)
# #     print (group.iloc[0])
#
#
