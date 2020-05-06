# Vector: 크기와 방향을 가진 양
#       : 1차원 배열/ 리스트로 표현
# Matrix: 행과 열을 가진 2차원 형상
# Tensor: 3차원 이상

# Tensor
import numpy as np
# 1) 0D Tensor
# 스칼라 즉 하나의 실수값으로 이뤄진 데이터.
d = np.array(5)
print(d.ndim)  # 0
print(d.shape)  # ()

# 2) 1D Tensor
# 벡터
d = np.array([1, 2, 3, 4])
print(d.ndim)  # 1
print(d.shape)  # (4,) 벡터개념에선 4차원이지만, 텐서에서는 1차원

# 3) 2D Tensor
# 행렬
d=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(d.ndim)  # 2
print(d.shape)  # (3, 4)

# 4) 3D Tensor
# 3차원 이상의 텐서부터 본격적으로 텐서라고 통칭함.
# ( batch_size x timesteps x word_dim )
# samples/batch_size: 데이터의 개수
# timesteps: 시퀀스의 길이
# word_dim: 단어를 표현하는 벡터의 차원
d = np.array([
    [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [10, 11, 12, 13, 14]
    ],
    [
        [15, 16, 17, 18, 19],
        [19, 20, 21, 22, 23],
        [23, 24, 25, 26, 27]
    ]
])
print(d.ndim)  # 3
print(d.shape)  # (2, 3, 5)


# 벡터와 행렬의 연산
a = np.array([1, 3, 2])
b = np.array([8, 9, 0])
print(a + b)  # [ 9 12  2]
print(b - a)  # [ 7  6 -2]

a = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
b = np.array([[5, 6, 7, 8], [1, 2, 3, 4]])
print(a + b)  # [[15 26 37 48] [51 62 73 84]]
print(b - a)  # [[ -5 -14 -23 -32] [-49 -58 -67 -76]]

# 벡터의 내적(점곱)
# 같은 차원의 "행벡터·열벡터" 여야 성립 가능. 결과는 스칼라가 됨.
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))  # 32

# 행렬의 곱셈
# 좌측 행렬의 행벡터 x 우측 행렬의 열벡터의 내적이 결과 행렬의 원소가 됨.
# 좌측행렬의 행벡터는 행의 위치를, 우측행렬의 열벡터는 열의 위치를 지정하게 됨.
a = np.array([[1, 4],
              [2, 4]])
b = np.array([[5, 3],
              [6, 7]])
print(np.matmul(a, b))  # [[29 31] [34 34]]
