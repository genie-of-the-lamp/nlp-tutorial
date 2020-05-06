# 다중 입력에 대한 실습

# 다중 선형 회귀(Multiple Linear Regression)
# 3개의 점수로 환산되는 최종 성적
#
# Midterm(x1)	|	Final(x2)	|	Added point(x3)	|	Score($1000)(y)
# ---------------------------------------------------------------------
# 70	     	|   85      	|	11          	|	73
# 71	     	|   89      	|	18          	|	82
# 50	     	|   80      	|	20          	|	72
# 99	     	|   20      	|	10          	|	57
# 50	     	|   10      	|	10          	|	34
# 20	     	|   99      	|	10          	|	58
# 40	     	|   50      	|	20          	|	56
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

X = np.array([[70, 85, 11], [71, 89, 18], [50, 80, 20], [99, 20, 10], [50, 10, 10]])
y = np.array([73, 82, 72, 57, 34])

model = Sequential()
model.add(Dense(1, input_dim=3,  # x값 3개
                activation='linear'))
sgd = optimizers.SGD(lr=0.00001)
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
model.fit(X, y, batch_size=1, epochs=20000, shuffle=False)

print(model.predict(X))

X_test = np.array([[20, 99, 10], [40, 50, 20]])  # 실제값은 각각 58, 56
print(model.predict(X_test))  # [[57.885456] [56.052002]], 근사치가 예측됨


# 다중 로지스틱 회귀
# OR GATE : 0 또는 1을 입력 받는데,
# 두 개의 입력이 모두 0일 때만 0 이외엔 모두 1 출력
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))
model.compile(optimizer='sgd', loss='binary_crossentropy',
              metrics=['binary_accuracy'])
model.fit(X, y, batch_size=1, epochs=800, shuffle=False)

print(model.predict(X))