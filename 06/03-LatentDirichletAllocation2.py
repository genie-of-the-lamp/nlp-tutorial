# 잠재 디리클레 할당 실습2
# 사이킷런을 사용한 LDA 수행
import pandas as pd
# import urllib.request
# urllib.request.urlretrieve("https://raw.githubusercontent.com/franciscadias/data/master/abcnews-date-text.csv",
#                            filename="./data/abcnews-date-text.csv")
data = pd.read_csv('./data/abcnews-date-text.csv', error_bad_lines=False)

print(len(data))  # 1,082,168개의 샘플 데이터.
text = data[['headline_text']]

#데이터 전처리
import nltk
text['headline_text'] = text.apply(lambda row: nltk.word_tokenize(row['headline_text']),
                                   axis=1  # axis=0 : row, axis=1 : column
                                   )  # 단어 토큰화
from nltk.corpus import stopwords
stop = stopwords.words('english')
text['headline_text'] = text['headline_text'].apply(
    lambda x: [word for word in x if word not in (stop)])  # 불용어 제거

from nltk.stem import WordNetLemmatizer
text['headline_text'] = text['headline_text'].apply(
    lambda x: [WordNetLemmatizer().lemmatize(word, pos="v")
               for word in x])  # 표제어 추출(인칭 통일, 시점 통일)

tokenized_doc = text['headline_text'].apply(
    lambda x: [word for word in x if len(word) > 3])  # 길이 3이하의 단어 제거
detokenized_doc = [" ".join(doc) for doc in tokenized_doc]
text['headline_text'] = detokenized_doc  # 사이킷런 함수 파라미터 형식에 맞게 역토큰화

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english',
                             max_features=1000)  # 상위 1000개
X = vectorizer.fit_transform(text['headline_text'])
print(X.shape)  # (1082168, 1000), 1,082,168 x 1,000 크기의 TF-IDF행렬

# 토픽 모델링
from sklearn.decomposition import LatentDirichletAllocation
ldamodel = LatentDirichletAllocation(n_components=10,
                                     learning_method='online',
                                     random_state=777,
                                     max_iter=1)

lda_top = ldamodel.fit_transform(X)
print(ldamodel.components_)  # VT
print(ldamodel.components_.shape)  # (10, 1000) 10개의 토픽 x 1000개의 단어

terms = vectorizer.get_feature_names()  # 단어 집합.

def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print("Topic {}".format(idx+1),
              [(feature_names[i], topic[i].round(2))
               for i in topic.argsort()[: -n - 1: -1]])

get_topics(ldamodel.components_, terms)  # 토픽별 상위 5개 분포 단어 출력