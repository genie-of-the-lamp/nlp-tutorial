#-*- coding:utf-8 -*-

# Bag of Words(BoW)
# 단어의 순서를 고려하지 않고 출현 빈도에만 집중하는 텍스트 데이터의 수치화 표현 방법.
# 1) 각 단어에 고유한 정수 인덱스 부여.
# 2) 인덱스의 위치에 단어 토큰의 등장 횟수를 기록한 벡터 생성.
# 주로 분류 문제나 여러 문서 간의 유사도 도출에 사용.

from konlpy.tag import Okt
import re
okt = Okt()

doc1 = "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다."


def exe_bow(doc):
    token = re.sub("(\.)", "", doc)  # 온점 제거
    token = okt.morphs(token)  # 토큰화

    word2index = {}
    bow = []
    for voca in token:
        if voca not in word2index.keys():
            word2index[voca] = len(word2index)
            bow.insert(len(word2index) - 1, 1)
        else:
            index = word2index.get(voca)
            bow[index] = bow[index] + 1
    return word2index, bow

result = exe_bow(doc1)
print(result[0])
print(result[1])

doc2 = "소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다."
result = exe_bow(doc2)
print(result[0])
print(result[1])

doc3 = doc1 + doc2
result = exe_bow(doc3)
print(result[0])
print(result[1])

# CountVectorizer 클래스로 Bow 만들기.
# 사이킷 런에서 지원하는 클래스로 단어의 빈도를 Count하여 Vector로 만듦.
# 띄어쓰기만을 기준으로 자르는 낮은 수준의 토큰화를 진항하기 때문에
# 한국어 사용에는 조사 등의 이유로 제대로 BoW를 생성할 수 없음.
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['you know I want your love. because I love you.']
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray())  # 코퍼스로부터 각 단어의 빈도수 기록
print(vector.vocabulary_)

text = ["Family is not an important thing. It's everything."]
vector = CountVectorizer(stop_words=["the", "a", "an", "is", "not"])  # 불용어 지정
print(vector.fit_transform(text).toarray())
print(vector.vocabulary_)

vector = CountVectorizer(stop_words="english")  # 자체 지원 불용어 사용
print(vector.fit_transform(text).toarray())
print(vector.vocabulary_)

from nltk.corpus import stopwords
vector = CountVectorizer(stop_words=stopwords.words("english"))  # nltk에서 지원하는 불용어 사용
print(vector.fit_transform(text).toarray())
print(vector.vocabulary_)
