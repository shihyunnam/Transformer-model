import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf

#to make our own layer
#Positional Encoding

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        #즉, Layer 클래스의 모든 기본 설정을 PositionalEncoding 클래스에도 적용, PositionalEncoding이 Layer의 모든 속성과 메소드를 상속하게
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    def get_angles(self, position, i, d_model):
        print("position shape is ", position.shape)
        print(i.shape)
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32)) # 1 / ((10000 * (2* i를 2로 나눈후의 몫)(홀짝구분)/ d_model 실수형으로))
        return position * angles
    def positional_encoding(self, position, d_model):
        
        angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],#ex) [:, tf.newaxis] makes 1d to 2d if it was 1d vector: [0, 1, 2, 3]은 [[0], [1], [2], [3]],문장에서 각 단어의 위치를 나타내는 숫자들을 만들어요. 이 숫자들은 각 단어가 문장의 어디에 있는지 
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],# [0, 1, 2, 3]은 [[0, 1, 2, 3]]으로 변환 #각 단어를 나타내는 숫자들의 위치
        d_model=d_model)
        print("angle_rads shape is ", angle_rads.shape)

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
print("Positional encoding started")
sample_pos_encoding = PositionalEncoding(50, 128)
print("Positional encoding ended")

plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 128))
plt.ylabel('Position')
plt.colorbar()
plt.show()

#Sclaed_dot_product_attention
def scaled_dot_product_attention(query, key, value, mask):
    # query 크기 : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    # key 크기 : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
    # value 크기 : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
    # padding_mask : (batch_size, 1, 1, key의 문장 길이)
    matmul_qk = tf.matmul(query, key, transpose_b = True)
    #key벡터의 차원수
    Dk = tf.cast(tf.shape(key)[-1], tf.float32)#key 벡터의 차원을 얻어서마지막 요소를 가져와서 int -> float으로 
    logits = matmul_qk / tf.math.sqrt(Dk)
    # 마스킹. 어텐션 스코어 행렬의 마스킹 할 위치에 매우 작은 음수값을 넣는다.
    # 매우 작은 값이므로 소프트맥스 함수를 지나면 행렬의 해당 위치의 값은 0이 된다.
    if mask is not None:
        logits += (mask * -1e9)
    attention_weight = tf.nn.softmax(logits, axis= -1)#행에 대해서 소프트맥스 적용 #행의 모든값의 합이 1이된다.
    output = tf.matmul(attention_weight, value)
    return output , attention_weight

# 임의의 Query, Key, Value인 Q, K, V 행렬 생성
np.set_printoptions(suppress=True)
temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
temp_k = tf.constant([[10,0,0],
                      [0,10,0],
                      [0,0,10],
                      [0,0,10]], dtype=tf.float32)  # (4, 3)

temp_v = tf.constant([[   1,0],
                      [  10,0],
                      [ 100,5],
                      [1000,6]], dtype=tf.float32)  # (4, 2)

# 함수 실행
#Q * KT = (1,4) -> scaling -> *v = (1,2)
temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
print(temp_attn) # 어텐션 분포(어텐션 가중치의 나열)
print(temp_out) # 어텐션 값


temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)
temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
print(temp_attn) # 어텐션 분포(어텐션 가중치의 나열)
print(temp_out) # 어텐션 값
