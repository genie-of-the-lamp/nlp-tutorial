# 코사인 유사도 Cosine Similarity
# 두 벡터 간의 코사인 각도를 이용하여 구할 수 있는 두 벡터의 유사도
# 방향이 완전히 동일할 경우 1, 90도의 각을 이루면 0
# 180도로 반대의 방향을 가지면 -1의 값을 가짐.
# 1에 가까울 수록 유사도가 높다고 판단.

# Numpy를 이용한 코사인 유사도 도출
from numpy import dot
from numpy.linalg import norm
import numpy as np

def cos_sim(A, B):
    return dot(A, B)/(norm(A) * norm(B))

doc1 = np.array([0,1,1,1])
doc2 = np.array([1,0,1,1])
doc3 = np.array([2,0,2,2])

print(cos_sim(doc1, doc2))  # 0.6666666666666667
print(cos_sim(doc1, doc3))  # 0.6666666666666667
print(cos_sim(doc2, doc3))  # 1.0000000000000002

# 유사도를 이용한 추천 시스템
# https://www.kaggle.com/rounakbanik/the-movies-dataset
import os
import pandas as pd
this_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(this_dir, 'meta', 'movies_metadata.csv'), low_memory=False)
data = data.head(20000)  # 20000개의 샘플로 학습
print(data['overview'].isnull().sum())  # 135개의 Null 데이터
data['overview'] = data['overview'].fillna('')  # Null을 빈 값으로 대체하여 예외 처리

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['overview'])
print(tfidf_matrix.shape)  # (20000, 47487) : 20000개의 영화 리뷰에 총 47,487개의 단어가 사용됨.

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(data.index, index=data['title']).drop_duplicates()
print(indices.head())
idx = indices['Father of the Bride Part II']
print(idx)  # 타이틀에 대한 인덱스를 반환하는 테이블

def get_recommendations(title, consine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))  # 모든 영화에 대한 유사도
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # 유사도에 따라 정렬
    sim_scores = sim_scores[1:11]  # 가장 유사한 10개
    movie_indices = [i[0] for i in sim_scores]

    return data['title'].iloc[movie_indices]

# 다크나이트 라이즈와 overview가 유사한 영화
print(get_recommendations('The Dark Knight Rises'))
