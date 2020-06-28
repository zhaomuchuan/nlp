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

# nltk.download('stopwords')
# stopwords.words('english')
#
response = requests.get('http://10.25.130.230:8080/about.html')

html = response.text

soup = BeautifulSoup(html,'html.parser')

text = soup.get_text(strip=True)

# 分句
X_Org = sent_tokenize(text,'english')
# print(X_Org)

# 矢量化
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_Org)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)
weight = tfidf.toarray()

# 聚类
kmeans = KMeans(n_clusters=3).fit(weight)
labels = kmeans.labels_
result = DataFrame({'category':labels,'sentence':X_Org})
grouped = result.groupby('category')
for name,group in grouped:
    print (name)
    print (group.iloc[0])


