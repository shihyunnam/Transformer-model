#Deep learning <- data 
#keras <- tensorflow를 사용하기 쉽게하기 위한 라이브러리
import keras
from keras.models import Sequential #기본적인 인공 신경망은 레이어가 순차적으로 구성되어있다.시퀀셜함수
#dense를 사용해 각레이어의 뉴런수설정가능, activation -> layer 사이에 활성화함수
from keras.layers import Dense, Activation # Dense -> fully connected layer(전결합층) layer examples -> 입력층, 은닉층.. 층들이 앞 층들과 연결되어있는것
from keras.utils import to_categorical#원핫인코딩구현 함수
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
#xtrain->학습해야할데이터, ytrain->학습해야할데이터의 정답
(x_train, y_train), (x_test, y_test) = mnist.load_data() # train 딥러닝을 학습시키는 데이터
print("x_train shape", x_train.shape)#(60000, 28, 28)28,28의 픽셀의 이미지로 처리됨
print("y_train shape", y_train.shape)
print("x_test shape", x_test.shape)
print("y_test shape", y_test.shape)
#mnist data의  x형태를 1차원으로 변경
x_train = x_train.reshape(60000, 784)#28*28 1*784형태로 바뀜
x_test = x_test.reshape(10000, 784)#28*28
x_train = x_train.astype("float32")#data normalization 0 to 1 values
x_train /= 255#black 0 , white 1
x_test = x_test.astype("float32")#data normalization 0 to 1 values
x_test /= 255#black 0 , white 1
print("x_train revised shape", x_train.shape)
print("x_test revised shape", x_test.shape)


#mnist data의  y형태를 변경 수치화->범주형으로
y_train = to_categorical(y_train, 10)# [0,0,0,0,0,0,0,1,0,0]형태로
y_test = to_categorical(y_test, 10)
print("y_train shape", y_train.shape)
print("y_test shape", y_test.shape)

#인공지능 모델 설계
#입력층 -> 은닉층들 -> 출력층 점점 개수가 줄어듬
#ReLU의 주요 목적은 음수 값을 0으로 만드는 것이므로, 0과 1 사이의 값에는 영향을 미치지 않습니다.

dlmodel = Sequential()
#모델에 층추가, 첫번째파래미터 :첫번쨰은닉층노드개수
dlmodel.add(Dense(512, input_shape= (784,)))
dlmodel.add(Activation('relu'))
dlmodel.add(Dense(256))
dlmodel.add(Activation('relu'))
dlmodel.add(Dense(10))
dlmodel.add(Activation('softmax'))#sum is 1
dlmodel.summary()

#parameter = 가중치 갯수(노드와 노드사이의 선가중치)

#모델 학습, 신경망을 잘 학습시키기 위해서는 학습한 신경망이 분류한값과 실제값의 오차를 계산, using gradient descent to 오차를 줄이려고
dlmodel.compile(loss = "categorical_crossentropy", optimizer='adam', metrics=['accuracy'])#metrics->평가지표
dlmodel.fit(x_train,y_train,batch_size=128, epochs=10, verbose=1)


#모델정확도
#검증데이터를통해서
score = dlmodel.evaluate(x_test, y_test)
print("test_score", score[0])#score[0] -> 오차값 0부터1, 0에 가까울수록 오차가 적다
print("test_score", score[1])#score[1] -> 정확도 0부터1
#모델학습결과확인하기
# predicted_class = np.argmax(dlmodel.predict(x_test), axis = 1)#10개의 값들중 가장큰값이 정답이므로(가로,행)
# # print(predicted_class.shape)

# correct_indices = np.nonzero(predicted_class == y_test)[0]
# incorrect_indices = np.nonzero(predicted_class != y_test[1])[0]


# 모델의 예측 결과
predicted_class = np.argmax(dlmodel.predict(x_test), axis=1)

# y_test에서 가장 높은 값을 가진 인덱스를 찾아서 y_test_class 배열 생성
y_test_class = np.argmax(y_test, axis=1)

# 예측이 정확한 인덱스
correct_indices = np.nonzero(predicted_class == y_test_class)[0]

# 예측이 부정확한 인덱스
incorrect_indices = np.nonzero(predicted_class != y_test_class)[0]

# 결과 출력
print("Correctly predicted indices:", correct_indices)
print("Incorrectly predicted indices:", incorrect_indices)


import matplotlib.pyplot as plt

# 잘못 예측된 첫 10개 이미지를 시각화
plt.figure(figsize=(10, 4))
for i, incorrect in enumerate(incorrect_indices[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_class[incorrect], y_test_class[incorrect]))
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()
