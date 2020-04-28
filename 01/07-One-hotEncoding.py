#-*- coding:utf-8 -*-

# Vocabulary: 단어집합 or 사전 : 서로 다른 단어들의 집합. book, books와 같은 경우도 포함.
#                               One-hot Encoding에 있어 먼저 생성되어야 함.


# One-hot Encoding: 자연어 처리를 위해 문자를 숫자로 바꾸는 여러 기법 중,
#                   가장 기본적인 표현 방법.
#                   단어 집합의 크기를 벡터의 차원으로 하고,
#                   표현하고 싶은 단어의 인덱스에 1을 다른 인덱스에는 0을 부여하는 벡터 표현 방식
#                       -> One-hot Vector

from konlpy.tag import Okt
okt = Okt()

# vocabulary
token = okt.morphs("나는 자연어 처리를 배운다")
print(token)

# indexing
word2index = {}
for voca in token:
    if voca not in word2index.keys():
        word2index[voca] = len(word2index)
print(word2index)

# set vector expression
def one_hot_encoding(word, word2index):
    one_hot_vector = [0]*(len(word2index));
    index = word2index[word]
    one_hot_vector[index] = 1
    return one_hot_vector

vector = one_hot_encoding("자연어", word2index)
print(vector)


# Keras를 이용한 One-hot Encoding
# to_categorical() : Keras에서 지원하는 One-hot Encoding Tool
text = "어제 같이 먹은 돼지 국밥 또 먹고 싶다 같이 먹으러 가실 분 또 먹으러 가면 돼지인가요"

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

t = Tokenizer()
t.fit_on_texts([text])
print(t.word_index)  # 정수 인코딩 결과

sub_text = "같이 먹으러 가실 분"; # 위 tokenizer에 포함된 vocabulary의 단어로만 존재해야 함.
encoded = t.texts_to_sequences([sub_text])[0]
print(encoded)

one_hot = to_categorical(encoded)  # '같이 먹으러 가실 분' 에 대한 One-hot Encoding
# [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]    index 1 vector
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]    index 3 vector
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]    index 10 vector
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]   index 11 vector
print(one_hot)


# One-hot Encoding의 한계
#   1) 단어의 개수 증가 -> 벡터 저장에 필요한 공간(벡터의 차원) 증가
#   2) 단어의 유사도 표현 불가능: 검색 시스템 등에서의 치명적 문제
#       - 해결 방법: 카운트 기반 벡터화 (LSA, HAL 등)
#                   예측 기반 벡터화 (NNLM, RNNLM, Word2Vec, FastText 등)
#                   둘 다 사용하는 GloVe

