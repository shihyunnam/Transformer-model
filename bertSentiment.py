
##########
# Loading necessary library
import time
import tensorflow as tf
import tensorflow_datasets as tfds
from IPython.display import clear_output
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
#divide train and testing dataset, load data for supervised learning,
# Loading imbd review dataset from tensorflow dataset
(ds_train, ds_test), ds_info = tfds.load('imdb_reviews',split = (tfds.Split.TRAIN, tfds.Split.TEST),
          as_supervised=True,with_info=True)
clear_output()

# initializing the tokenizer -> 모델이 처리할수 있는 형태로 변환시킴, using hugging face transformer library
#uncased -> 대소문자구별x  changing it to all lower case letters
# Loading imbd review dataset from tensorflow dataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
clear_output()

#리뷰 텍스트를 받아 BERT 모델에 적합한 형태로 변환하고, 변환된 정보를 반환하는 역할을 합니다. 주로 BERT 모델을 사용한 자연어 처리 작업에서 입력 데이터를 준비하는 데 사용됩니다.
def convert_example_to_feature(review):
    #BERT 모델이 처리할 수 있는 형식으로 인코딩
    return tokenizer.encode_plus(review,
                add_special_tokens = True, # add [CLS], [SEP] for start and the end of the sentences
                max_length = max_length, # max length of the text that can go to BERT
                pad_to_max_length = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
              )

# can be up to 512 for BERT
max_length = 512
batch_size = 6
#mapping into dictionary with input data
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label#label for target value


# prepare list, so that we can build up final TensorFlow dataset from slices, Preprocessing
def encode_examples(ds, limit=-1):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    if (limit > 0):
        ds = ds.take(limit)
    for review, label in tfds.as_numpy(ds):
        bert_input = convert_example_to_feature(review.decode())
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
        #change it to  TensorFlow dataset with decoded .
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)


# train dataset, 훈련 데이터셋 준비
start=time.time()
ds_train_encoded = encode_examples(ds_train).shuffle(10000).batch(batch_size)
print("Done with Training Dataset",time.time()-start)

# test dataset 테스트 데이터셋 준비
start=time.time()
ds_test_encoded = encode_examples(ds_test).batch(batch_size)
print("Done with Testing Dataset",time.time()-start)

# recommended learning rate for Adam 5e-5, 3e-5, 2e-5
learning_rate = 2e-5
# we will do just 1 epoch, though multiple epochs might be better as long 
# as we will not overfit the model
number_of_epochs = 1

# model initialization
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# choosing Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
# we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_test_encoded)
###TESTING##
# test_sentence = "This is a really good movie. I loved it and will watch again"
# predict_input = tokenizer.encode(test_sentence,truncation=True,padding=True,return_tensors="tf")

# tf_output = model.predict(predict_input)[0]
# tf_prediction = tf.nn.softmax(tf_output, axis=1)
# labels = ['Negative','Positive'] #(0:negative, 1:positive)
# label = tf.argmax(tf_prediction, axis=1)
# label = label.numpy()
# print(labels[label[0]])