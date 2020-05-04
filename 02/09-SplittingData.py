#-*- coding:utf-8 -*-

# Slitting Data
# 머신러닝 훈련 중 지도 학습을 위한 데이터 분리 작업.

# Supoervised Learning 지도학습
# 답이 정해진 학습 데이터를 통한 훈련 향후에 답이 없는 문제에도 정답을 도출해낼 것을 목표로 함
# x_train: 문제 데이터, y_train: 문제 정답 데이터 -> 문제와 정답을 동시에 조회하여 규칙을 도출
# x_test: 시험 데이터 y_test: 시험 정답 데이터 -> 정답 없이 문제만 주어주고 그 정답 수치로 정확도를 매김

# X와 y 분리
# 1) zip()
X, y = zip(['a', 1], ['b', 2], ['c', 3])
print(X)
print(y)


# 2) 데이터 프레임
import pandas as pd
values = [['당신에게 드리는 마지막 혜택!', 1],
['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨. 잘 지내시죠? 오랜만입...', 0],
['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns = ['메일 본문', '스팸 메일 유무']

df = pd.DataFrame(values, columns=columns)
print(df)

X=df['메일 본문']
y=df['스팸 메일 유무']
print(X)
print(y)


# 3) Numpy
import numpy as np
ar = np.arange(0, 16).reshape((4,4))
print(ar)

X = ar[:, :3]
y = ar[:,3]
print(X)
print(y)



# 테스트 데이터 분리
# 사이킷런
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)
# X : 독립 변수 데이터. (배열이나 데이터프레임)
# y : 종속 변수 데이터. 레이블 데이터.
# test_size : 테스트용 데이터 개수를 지정한다. 1보다 작은 실수를 기재할 경우, 비율을 나타낸다.
# train_size : 학습용 데이터의 개수를 지정한다. 1보다 작은 실수를 기재할 경우, 비율을 나타낸다.
# (test_size와 train_size 중 하나만 기재해도 가능)
# random_state : 난수 시드

import numpy as np
from sklearn.model_selection import train_test_split
X, y = np.arange(10).reshape((5, 2)), range(5)
print(X)
print(list(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)  # 33%만 테스트 데이터로 지정
print(X_train)
print(X_test)
print(y_train)
print(y_test)

