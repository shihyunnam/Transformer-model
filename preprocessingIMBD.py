from collections import Counter
import math
import os
import random
from torch.utils.data import Dataset, DataLoader
import torch
#imdb dataset 정리하기
def load_imdb_dataset(imdb_data_path, num_samples=None):
    #datasets, train texts, labels, test texts, labels
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    # Directories for train and test data
    train_dir = os.path.join(imdb_data_path, 'train')
    test_dir = os.path.join(imdb_data_path, 'test')

    # Loading the training data
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname)) as f:
                    train_texts.append(f.read())
                train_labels.append(0 if label_type == 'neg' else 1)

    # Loading the test data
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(test_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname)) as f:
                    test_texts.append(f.read())
                test_labels.append(0 if label_type == 'neg' else 1)
    # Optionally, shuffle and limit the number of samples
    if num_samples:
        #pair text and label together
        combined = list(zip(train_texts, train_labels))
        #shuffle the elements
        random.shuffle(combined)
        train_texts[:], train_labels[:] = zip(*combined[:num_samples])    #분리

        combined = list(zip(test_texts, test_labels))
        random.shuffle(combined)
        test_texts[:], test_labels[:] = zip(*combined[:num_samples])

    return (train_texts, train_labels), (test_texts, test_labels)

# Path to the IMDb dataset
imdb_data_path = '/Users/shihyunnam/Downloads/aclImdb'  # Replace with the path to the downloaded IMDb dataset
(train_texts, train_labels), (test_texts, test_labels) = load_imdb_dataset(imdb_data_path, num_samples=1000)  # Load a subset for quicker experimentation
# Print the size of train_texts and train_labels
# print("Size of train_texts:", len(train_texts))
# print("Size of train_labels:", len(train_labels))

# # Print the first 5 rows of the dataset
# for i in range(50):
#     print(f"Review {i+1}: {train_texts[i]}")
#     print(f"Label {i+1}: {'Positive' if train_labels[i] == 1 else 'Negative'}")
#     print("---")

def tokenize(texts):
    return [text.split() for text in texts]

from collections import Counter
def build_vocab(texts):
    # 모든 리뷰에서 단어들의 빈도수를 계산합니다
    counter = Counter(word for text in texts for word in text)
    # 가장 빈번한 단어부터 시작하여 단어 사전을 구축합니다
    return {word: i + 2 for i, (word, _) in enumerate(counter.most_common())}

def encode_texts(texts, vocab):
    return [[vocab.get(word, 1) for word in text] for text in texts]  # 1은 <unk>를 나타냅니다
def pad_sequences(sequences, max_len):
    return [seq[:max_len] + [0] * max(0, max_len - len(seq)) for seq in sequences]  # 0은 패딩 토큰입니다
# 구두점 제거 적용
import string
def preprocess_text(text):
    # 구두점 제거 및 소문자화
    text = text.lower()  # 소문자화
    text = text.translate(str.maketrans('', '', string.punctuation))  # 구두점 제거
    return text

# 텍스트 전처리 (구두점 제거 및 소문자화 적용)
train_texts_processed = [preprocess_text(text) for text in train_texts]
test_texts_processed = [preprocess_text(text) for text in test_texts]

# 토큰화
train_texts_tokenized = tokenize(train_texts_processed)
test_texts_tokenized = tokenize(test_texts_processed)

# 단어 사전 구축
vocab = build_vocab(train_texts_tokenized)
print(vocab)
# 텍스트를 정수 시퀀스로 변환
train_sequences = encode_texts(train_texts_tokenized, vocab)
test_sequences = encode_texts(test_texts_tokenized, vocab)

# 패딩 적용
max_seq_length = 128  # 최대 시퀀스 길이 설정
train_sequences_padded = pad_sequences(train_sequences, max_seq_length)
test_sequences_padded = pad_sequences(test_sequences, max_seq_length)

print("here2")
###################2

class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# DataLoader 생성
train_dataset = IMDBDataset(train_sequences_padded, train_labels)
test_dataset = IMDBDataset(test_sequences_padded, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#############3 model definition

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerForSentimentAnalysis(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, max_seq_length=128, num_classes=2):
        super(TransformerForSentimentAnalysis, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        out = memory.mean(dim=1)
        out = self.fc_out(out)
        return out


# print("here3")
def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# print("here4")

def evaluate(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
print("here5")
# 손실 함수 및 옵티마이저 설정
# 모델 인스턴스 생성
vocab_size = len(vocab) + 2  # 단어 사전 크기 + 2 (특수 토큰을 위한 공간)
model = TransformerForSentimentAnalysis(vocab_size=vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 에포크 수 설정
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    test_accuracy = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Test Accuracy: {test_accuracy}")

