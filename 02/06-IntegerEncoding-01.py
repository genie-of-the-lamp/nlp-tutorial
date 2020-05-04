#-*- coding:utf-8 -*-

# NLTK의 FreqDist를 사용한 정수 인코딩

import os
os.path.join(r'C:\Users\genie.jung\Anaconda3\Library\bin');

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
import numpy as np
text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."
# 문장 토큰화
text = sent_tokenize(text)

vocab = {}
sentences = []
stop_words = set(stopwords.words('english')) # 불용어 리스트

for i in text:
    sentence = word_tokenize(i)  # 단어 토큰화
    result = []

    for word in sentence:
        word = word.lower()  # 소문자로 통일해 중복 제거
        if word not in stop_words:  # 불용어 제거
            if len(word) > 2:  # 짧은 단어 제거
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0
                vocab[word] += 1
    sentences.append(result)
    
# np.hstack -> 문장을 구분하는 depth제거
# FreqDist 빈도수 계산 도구
vocab = FreqDist(np.hstack(sentences))

vocab_size = 5
vocab = vocab.most_common(vocab_size)  # 빈도 수가 높은 상위 5개 까지만 vocab에 담음

word_to_index = {word[0]: index + 1 for index, word in enumerate(vocab)}

