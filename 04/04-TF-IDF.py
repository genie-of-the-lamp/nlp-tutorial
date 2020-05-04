# Document-Term Matrix 문서 단어 행렬 (DTM)
# 서로 다른 문서들의 BoW들을 결합한 표현 방식.

# TF-IDF(Term Frequency - Inverse Document Frequency)
# DTM 내의 각 단어에 대한 중요도를 계산하는 가중치.
# 주로 문서의 유사도를 구하는 작업, 검색 시스템에서 검색 결과의 중요도를 정하는 작업,
# 문서 내에서 특정 단어의 중요도를 구하는 작업 등에 쓰임.
# tf(d,t): 특정 문서 d에서 특정 단어 t의 등장 횟수.
# df(t): 특정 단어 t가 등장한 문서의 수.
# idf(d, t): df(t)에 반비례 하는 수.
#           - 기하급수적인 가중치 차를 대비하기 위해 log를 사용 (기본적으로 자연로그 사용)
#           - 등장 문서 수가 0일 경우 분모가 0이 되는 것을 방지하기 위해 분모에 1을 더함.

import pandas as pd  # 데이터 프레임 사용
from math import log  # IDF 계산

docs = [
    '먹고 싶은 사과',
    '먹고 싶은 바나나',
    '길고 노란 바나나 바나나',
    '저는 과일이 좋아요'
]

vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()

N = len(docs)  # 총 문서의 수

def tf(t, d):
    return d.count(t)

def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N/df + 1)

def tf_idf(t, d):
    return tf(t, d) * idf(t)

result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns=vocab)
print(tf_)

result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))

idf_ = pd.DataFrame(result, index=vocab, columns=["IDF"])
print(idf_)

result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tf_idf(t, d))

tf_idf_ = pd.DataFrame(result, columns=vocab)
print(tf_idf_)
# 전체 문서의 수 N과 df(t)가 1의 차이가 나게되면 idf(d,t)의 log 진수값이 1이되어
# idf(d,t) 값이 0이 되어 가중치의 역할을 하지 못하게 됨.(idf(d,t) = log(n/df(t)+1) = 0)
# 따라서 실제 구현체에서는 log항에 1을 더해주는 방법으로 IDF가 최소 1이상의 값을 갖게 함.

# 사이킷런을 이용한 DTM과 TF-IDF실습.
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',
]
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray())  # 코퍼스로부터 각 단어의 빈도수 기록
print(vector.vocabulary_)  # 인덱스 부여 상태

# TfidfVectorizer - 사이킷런에서 제공하는 TF-IDF를 자동 계산해주는 클래스
# 보편적 TF-IDF에서 좀 더 조정된 새로운 식을 사용함.
# (IDF의 로그항의 분자에 1을 더해주며, 로그항에 1을 더해주고, TF-IDF에 L2 정규화 적용 등)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)