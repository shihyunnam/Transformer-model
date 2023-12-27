import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf

#1.Limitation of previous seq2seq model:
#format of seq2seq model is a form of encoder-decoder model
#encoder: Part where it listens and understands the sentence: "저는 학교에 가요"
#decoder: gets the vector and creates a new sentence: "I go to school"
#problem: During the time when changing encoder into vector, some parts of the information could disappear so attention mechanism is used.
#Hyper parameter:Values that users can change when they design the model



#2.Hyper parameters for transformer
#Attention: focusing on important part in the sentence
#d_model = 512 : input, output size, word -> 512 size of vector
#num_layers = 6 
#num_heads = 8: parallel attention 
#d_ff = 2048 : 피드 포워드 신경망이 존재하며 해당 신경망의 은닉층의 크기, feed forward:데이터를 순차적으로 처리하는 계산의 연속

#3.Positional Encoding
#RNN was useful for NLP due to the advantage of having positional information
#Positional Encoding:트랜스포머는 단어의 위치 정보를 얻기 위해서 각 단어의 임베딩 벡터에 위치 정보들을 더하여 모델의 입력으로 사용

#to make our own layer
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        #즉, Layer 클래스의 모든 기본 설정을 PositionalEncoding 클래스에도 적용, PositionalEncoding이 Layer의 모든 속성과 메소드를 상속하게
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    def get_angles(self, position, i, d_model):
        print(position.shape)
        print(i.shape)
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32)) # 1 / ((10000 * (2* i를 2로 나눈후의 몫)(홀짝구분)/ d_model 실수형으로))
        return position * angles
    def positional_encoding(self, position, d_model):
        
        angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],#ex) [:, tf.newaxis] makes 1d to 2d if it was 1d vector: [0, 1, 2, 3]은 [[0], [1], [2], [3]],문장에서 각 단어의 위치를 나타내는 숫자들을 만들어요. 이 숫자들은 각 단어가 문장의 어디에 있는지 
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],# [0, 1, 2, 3]은 [[0, 1, 2, 3]]으로 변환 #각 단어를 나타내는 숫자들의 위치
        d_model=d_model)
        print(angle_rads.shape)

        #for even indices : 2i
        sines = tf.math.sin(angle_rads[:, 0::2])
        print(sines.shape)
        #for odd indices : 2i + 1
        cosines = tf.math.cos(angle_rads[:, 1::2])
        print(cosines.shape)

        print("angle_rads shape is ", angle_rads.shape)
        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        print("revised angle_rads shape is ", angle_rads.shape)
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        print(pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)

    #위치인코딩 적용함수
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
#입력 문장의 단어가 50개이면서, 각 단어가 128차원의 임베딩 벡터
sample_pos_encoding = PositionalEncoding(50, 128)
plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 128))
plt.ylabel('Position')
plt.colorbar()
plt.show()