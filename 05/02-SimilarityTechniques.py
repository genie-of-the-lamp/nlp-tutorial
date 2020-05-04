# 1. 유클리드 거리(Euclidean distance)
# 다차원 공간에서 두개의 점 p와 q가 각각의 좌표를 가질 때,
# 두 점 사이의 거리를 계산하는 방법.
# 여러 문서에 대해 유사도를 구하고자 할 때 유클리드 거리 공식을 사용한다는 것은,
# 단어의 총 개수만큼의 차원에서 문서 만큼의 점이 있는 것과 같음.
# 유클리드 거리의 값이 가장 적을 때 문서 간의 유사도가 가장 높음을 의미.
import numpy as np
def dist(x, y):
    return np.sqrt(np.sum((x-y)**2))

doc1 = np.array((2,3,0,1))
doc2 = np.array((1,2,3,1))
doc3 = np.array((2,1,2,2))
docQ = np.array((1,1,0,1))

print(dist(doc1, docQ))
print(dist(doc2, docQ))
print(dist(doc3, docQ))

# 2. 자카드 유사도(Jaccard similarity)
# 합집합에서 교집합의 비율을 구한다면 두 집합 사이의 유사도를 구할 수 있다는 아이디어.
# 0과 1사이의 값으로 동일하다면 1, 공통 원소가 없다면 0의 값을 가짐.
doc1 = "apple banana everyone like likey watch card holder"
doc2 = "apple banana coupon passport love you"

tokenized_doc1 = doc1.split()
tokenized_doc2 = doc2.split()

print(tokenized_doc1)
print(tokenized_doc2)

union = set(tokenized_doc1).union(set(tokenized_doc2))
print(union)  # 합집합

intersection = set(tokenized_doc1).intersection(set(tokenized_doc2))
print(intersection)  # 교집합

# 자카드 유사도 즉, 두 문서의 총 단어 집합에서 공통적으로 등장한 단어의 비율
print(len(intersection)/len(union))
