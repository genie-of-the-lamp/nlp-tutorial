# 잠재 의미 분석(Latent Semantic Analysis)
# DTM이나 TF-IDF가 단어의 의미를 고려하지 못한다는 단점의 대안으로
# DTM의 잠재된 의미를 이끌어내는 방법.
# DTM이나 TF-IDF 행렬에 절단된 SVD를 사용하여 차원을 축소시키고,
# 단어들의 잠재적 의미를 끌어낸다는 아이디어.

import numpy as np
A = np.array([[0, 0, 0, 1, 0, 1, 1, 0, 0],
              [0, 0, 0, 1, 1, 0, 1, 0, 0],
              [0, 1, 1, 0, 2, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 1, 1]])  # 4 x 9 크기의 DTM
print(np.shape(A))  # (4, 9)

U, s, VT = np.linalg.svd(A, full_matrices=True)  # Full SVD
print(U.round(2))
print(np.shape(U))  # (4, 4)
print(s.round(2))
print(np.shape(s))  # (4,) svd함수는 s로 대각행렬이 아닌 특이값 리스트를 반환함
S = np.zeros((4, 9))  # 대각행렬의 크기인 4 x 9의 임의의 행렬
S[:4, :4] = np.diag(s)  # 특이값 삽입
print(S.round(2))
print(np.shape(S))  # (4, 9)
print(VT.round(2))
print(np.shape(VT))  # (9, 9)

result = np.allclose(A, np.dot(np.dot(U, S), VT).round(2))
print(result)  # True 파라미터로 받는 두 행렬이 동일할 경우 True 반환

# Truncated SVD
S = S[:2, :2]
print(S.round(2))  # 상위 2개의 특이값만 남기고 제거
U = U[:, :2]
print(U.round(2))  # 2개의 열만 남기고 제거
VT = VT[:2, :]
print(VT.round(2))  # 2개의 행만 남기고 제거 (V행렬 관점에서는 2개의 열만 남은 것)
A_prime = np.dot(np.dot(U, S), VT)
print(A)
print(A_prime.round(2))  # 값을 소실시켰기 때문에 기존의 A 행렬로의 복구는 불가능함.

# U의 크기 : 문서의 수 x 토픽의 수
# U의 각 행은 잠재된 의미를 표현하기 위한 수치화된 각각의 문서 벡터를 의미.
# VT의 크기 : 토픽의 수 x 단어의 수
# VT의 각 열은 잠재의미를 표현하기 위해 수치화된 각각의 단어 벡터를 의미.


# LSA 실습
# 문서의 수를 원하는 토픽의 수로 압축한 뒤,
# 각 토픽당 가장 중요한 단어 5개를 출력하는 실습.
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers','quotes'))
documents = dataset.data
print(len(documents))  # 11314 훈련에 사용할 뉴스 수

# 텍스트 전처리
news_df = pd.DataFrame({'document': documents})
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ")  # 특수 문자 제거
news_df['clean_doc'] = news_df['clean_doc'].apply(
    lambda x: ' '.join([w for w in x.split() if len(w) > 3]))  # 3자 이하 단어 제거
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())  # 전체 소문자 통일

from nltk.corpus import stopwords
stop_words = stopwords.words('english')  # nltk에서 지원하는 영문 불용어
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])  #불용어 제거

detokenized_doc = []  # TfidVectorizer의 파라미터 형식에 맞게 역토큰화 진행
for doc in tokenized_doc:
    detokenized_doc.append(" ".join(doc))

news_df['clean_doc'] = detokenized_doc

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english',
                             max_features=1000,  # 상위 1,000개의 단어만 보존
                             max_df=0.5,
                             smooth_idf=True)
X = vectorizer.fit_transform(news_df['clean_doc'])
print(X.shape)  # (11314, 1000) 11,314 x 1,000의 크기를 가진 TF-IDF

# 토픽 모델링
# 뉴스 데이터의 카테고리 개수인 20을 토픽의 수로 가정
from sklearn.decomposition import TruncatedSVD
svd_model = TruncatedSVD(n_components=20,  # 토픽의 수
                         algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(X)
print(np.shape(svd_model.components_))  # (20, 1000) LSA의 VT에 해당

terms = vectorizer.get_feature_names()  # 단어 집합

def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print("Topic {}:".format(idx + 1),
              [(feature_names[i], topic[i].round(5))
               for i in topic.argsort()[: -n - 1: -1]])
        # argsort(): 크기의 오름차순 별 인덱스를 반환
        # 확장 슬라이스: 인덱싱에서 이중 콜론으로 표현. 마지막 인덱스가 확장방식을 결정함.
        #              정수값은 간격, -1은 역순의 리스트.

get_topics(svd_model.components_, terms)  # 각 20개 행의 각 1,000개의 열 중
                                          # 가장 값이 큰 5개의 값을 찾아 단어로 출력.

# 쉽고 빠른 구현과 잠재의미를 이끌어 낼 수 있는 LSA 지만
# 새로운 정보에 대한 업데이트가 어려운 단점을 가짐.
