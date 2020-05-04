#-*- coding:utf-8 -*-

# 케라스(Keras)의 텍스트 전처리

from tensorflow.keras.preprocessing.text import Tokenizer

sentences=[['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)  # 빈도수를 기준으로 단어 집합 생성

print(tokenizer.word_index)  # 정수 인코딩 (빈도수순 정렬, 인데스 부여)
print(tokenizer.word_counts)  # 빈도수 조회
print(tokenizer.texts_to_sequences(sentences))  # 단어 -> 인덱스 변환

# 사용 단어 수 지정
vocab_size = 5
tokenizer = Tokenizer(num_words= vocab_size + 1)  # num_words는 0부터 카운트
tokenizer.fit_on_texts(sentences)
print(u"\nnum_words 지정")
print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.texts_to_sequences(sentences))  # num_words는 texts_to_sequences에서만 적용됨 (word_index, word_counts는 X)

#다른 함수에서의 출력 값도 제한하고 싶을 경우
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = 5
words_frequenct = [w for w,c in tokenizer.word_index.items() if c >= vocab_size + 1]
for w in words_frequenct:
    del tokenizer.word_index[w]
    del tokenizer.word_counts[w]
print(u"\n직접 접근해 vocab_size에 해당하지 않는 단어 제거")
print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.texts_to_sequences(sentences))

# keras tokenizer는 OOV단어는 지워버림
# 필요 시 oov_token 사용
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 2, oov_token= "OOV") #OOV토큰이 추가되므로 +1이 아닌 +2
tokenizer.fit_on_texts(sentences)

print("OOV index : {}".format(tokenizer.word_index['OOV']))  # 01에서의 인덱싱과 달리 tf.keras tokenizer는 OOV 인덱스가 1
print(tokenizer.texts_to_sequences(sentences))
