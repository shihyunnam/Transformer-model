import csv


# File parsing
input_file_path = '/Users/shihyunnam/Desktop/Attention is All you need/parsed.csv'  # Update this to your input file path
output_file_path = '/Users/shihyunnam/Desktop/Attention is All you need/korean_sentences.csv'  # Update this to your desired output file path

def process_csv(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            if len(row) >= 2:#at least four columns
                # Extract and write the second and fourth columns
                writer.writerow([row[1]])

process_csv(input_file_path, output_file_path)



#data processing
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences


def build_vocab(sentences, min_freq=2):
    # Tokenize sentences and build vocabulary
    counter = Counter(token for sentence in sentences for token in sentence.split())
    vocab = {token for token, freq in counter.items() if freq >= min_freq}
    vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3, **{token: i + 4 for i, token in enumerate(vocab)}}
    return vocab

def sentence_to_indices(sentence, vocab, max_len):
    return [vocab.get(token, vocab['<unk>']) for token in sentence.split()][:max_len]

def pad_sentences(indices, max_len):
    return pad_sequences(indices, maxlen=max_len, padding='post', truncating='post', value=0)

# Building vocabularies
english_vocab = build_vocab(english_sentences)
korean_vocab = build_vocab(korean_sentences)

# Converting sentences to indices
max_len = 20  # or any other appropriate length
english_indices = [sentence_to_indices(sentence, english_vocab, max_len) for sentence in english_sentences]
korean_indices = [sentence_to_indices(sentence, korean_vocab, max_len) for sentence in korean_sentences]

# Padding sentences
english_indices = pad_sentences(english_indices, max_len)
korean_indices = pad_sentences(korean_indices, max_len)



