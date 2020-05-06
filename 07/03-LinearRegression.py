# 선형회귀(Linear Regression)
# 하나 이상의 독립변수 x와 종속변수 y 사이의 선형 관계 모델링
# if x가 하나 -> 단순 선형 회귀
# else -> 다중 선형 회귀

# 케라스로 구현하는 선형 회귀

# 케라스로 모델을 만드는 기본 형식
# model = keras.models.Sequential()  # Sequential 으로 모델 생성
# model.add(keras.layers.Dense(  # add 로 필요한 사항 추가
#     1,  # 출력의 차원
#     input_dim=1  # 입력의 차원
# ))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import numpy as np

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])  # 학습 시간
y = np.array([11, 22, 33, 44, 53, 66, 77, 87, 95])  # 성적

model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))  # 선형 회귀 지정
sgd = optimizers.SGD(lr=0.01)  # SGD: Stochastic Gradient Descent, lr: Learning Rate
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])  # mse: Mean Squared Error
model.fit(X, y, batch_size=1, epochs=300, shuffle=False)  # 훈련횟수 300

import matplotlib.pyplot as plt
plt.plot(X, model.predict(X), 'b', X, y, 'k.')
plt.show()  # graph 실행

print(model.predict([9.5]))  # [[98.556465]], 9.5의 학습 시간일 때의 성적 예측
