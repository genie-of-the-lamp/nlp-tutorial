# 잠재 디리클레 할당(Latent Dirichlet Allocation, LDA)
# 문서의 집합으로부터 어떤 토픽이 존재하는지를 알아내기 위한 알고리즘.
# 빈도수 기반 표현 방법(BoW, DTM, TF-IDF 행렬 등)을 입력으로 함. (즉 LDA는 단어의 순서를 신경쓰지 않음)
# 아래의 가정 하에 토픽 추출을 위해 그 과정을 역추적하는 역공학 수행.
# 역추적 과정에서의 토픽 개수 k는 하이퍼 파라미터이다.
# 하이퍼파라미터: 사용자가 직접 설정하는 매개변수

# LDA의 문서 작성에의 가정
# 1) 사용할 단어 개수 N 지정
# 2) 사용할 토픽의 혼합을 확률분포에 기반하여 결정
# 3-1) 토픽 분포에서 토픽 T를 확률적으로 선택
# 3-2) 선택한 토픽 T에서 단어의 출현 확률 분포에 기반해 문서에 사용할 단어 선택

# LSA: DTM을 차원 축소하여 축소차원에서 근접 단어들을 토픽으로 묶는다.
# LDA: 단어가 특정 토픽에 존재할 확률과 문서에 특정 토픽이 존재할 확률을 결합확률로 추정하여 토픽을 추출한다.

# gensim을 사용한 LDA 실습
# LSA에서 했던 전처리과정과 동일.
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
documents = dataset.data
news_df = pd.DataFrame({'document': documents})
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ")
news_df['clean_doc'] = news_df['clean_doc'].apply(
    lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

# 각 단어에 정수 인코딩과 동시에 뉴스 별 빈도수 기록, (word_id, word_frequency) 형태
from gensim import corpora
dictionary = corpora.Dictionary(tokenized_doc)
corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
print(corpus[1])  # [(0, 1), (2, 1), (20, 1), ...], index 1의 뉴스
print(dictionary[66])  # well, word_id 66의 단어
print(len(dictionary))  # 64365, 학습된 총 단어 수

# LDA 모델 훈련시키기
import gensim
NUM_TOPICS = 20  # 카테고리 수 만큼 토픽 개수 설정 (하이퍼 파라미터)
ldamodel = gensim.models.ldamodel.LdaModel(corpus,
                                           num_topics=NUM_TOPICS,
                                           id2word=dictionary,
                                           passes=15)  # 15회 알고리즘 반복
topics = ldamodel.print_topics(num_words=4)  # default=15
for topic in topics:
    print(topics)  # 단어 앞의 숫자는 해당 단어의 기여도를 나타냄.

def make_topictable_per_doc(ldamodel, corpus):  # 토픽 분포도 출력 함수
    topic_table = pd.DataFrame()

    for i, topic_list in enumerate(ldamodel[corpus]):  # 문서 별

        doc = topic_list[0] if ldamodel.per_word_topics else topic_list
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)  # 분포도가 높은 순 정렬

        for j, (topic_num, prop_topic) in enumerate(doc):
            if j == 0:
                topic_table = topic_table.append(pd.Series([int(topic_num),
                                                            # 가장 분포도가 큰 토픽 번호
                                                            round(prop_topic, 4),
                                                            # 분포도
                                                            topic_list
                                                            # 전체 토픽
                                                            ]),
                                                 ignore_index=True)
            else:
                break
    return topic_table

topictable = make_topictable_per_doc(ldamodel, corpus)
topictable = topictable.reset_index()  # 문서 번호 인덱싱
topictable.columns = ['문서 번호', '가장 비중이 높은 토픽',
                      '가장 높은 토픽의 비중', '각 토픽의 비중']
print(topictable)