# 로지스틱 회귀(Logistic Regression)
# 이진 분류 문제를 해결하는 대표적 알고리즘.

# 시그모이드 함수(Sigmoid function)
# 이진 분류문제에 적합하지 않은 선형 회귀의 문제점(0,1사이의 값, S자 형태의 그래프)들을
# 해결할 수 있어 로지스틱 회귀의 가설이 되는 함수
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))


# weight에 따른 그래프 변화  확인.
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5*x)  # 가중치 W가 0.5, bias가 0일 때
y2 = sigmoid(x)  # 가중치 W가 1, bias가 0일 때
y3 = sigmoid(2*x)  # 가중치 W가 2, bias가 0일 때

plt.plot(x, y1, 'r', linestyle='--')
plt.plot(x, y2)
plt.plot(x, y3, 'g', linestyle='--')
plt.plot([0,0], [1.0, 0.0], ":")  # 중앙 점선
plt.title('Sigmoid Function')
plt.show()  # W가 그래프의 경사도에 영향을 미침을 알 수 있음. 작을 수록 경사가 완만.

# bias에 따른 그래프 변화 확인.
y1 = sigmoid(x+0.5)
y2 = sigmoid(x+1)
y3 = sigmoid(x+1.5)

plt.plot(x, y1, 'r', linestyle='--')
plt.plot(x, y2)
plt.plot(x, y3, 'g', linestyle='--')
plt.show()  # b가 클 수록 1에 가까워짐.


# 케라스로 구현하는 로지스틱 회귀
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

X = np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

model = Sequential()
model.add(Dense(1, input_dim=1, activation='sigmoid'))  # 가설: 시그모이드 함수
sgd = optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd, loss='binary_crossentropy',  # 비용함수: 이진크로스엔트로피
              metrics=['binary_accuracy'])
model.fit(X, y, batch_size=1, epochs=200, shuffle=False)

plt.plot(X, model.predict(X), 'b', X, y, 'k.')  # 오차가 최소화된 W, b의 시그모이드 그래프
plt.show()

print(model.predict([1, 2, 3, 4, 4.5]))
print(model.predict([11, 21, 31, 41, 500]))
