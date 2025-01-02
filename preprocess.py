import os
import pickle
import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

trim_counter = {
    'wuxujieduan': 0,
    'qianjvhao': 0,
    'houjvhao': 0,
    'jiandanjieduan': 0
}

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+', '', text)
    return text

def trim_text(text, target_len=1000, counter=trim_counter):
    if len(text) <= target_len:
        counter['wuxujieduan'] += 1
        return text
    before_index = text.rfind('.', 0, target_len)
    after_index = text.find('.', target_len)
    if before_index == -1 and after_index == -1:
        counter['jiandanjieduan'] += 1
        return text[:target_len]
    elif before_index != -1 and after_index == -1:
        counter['qianjvhao'] += 1
        return text[:before_index + 1]
    elif before_index == -1 and after_index != -1:
        counter['houjvhao'] += 1
        return text[:after_index + 1]
    else:
        if target_len - before_index <= after_index - target_len:
            counter['qianjvhao'] += 1
            return text[:before_index + 1]
        else:
            counter['houjvhao'] += 1
            return text[:after_index + 1]

def preprocess_data(data_path, tokenizer_file_path, preprocessed_data_file_path):
    if os.path.exists(tokenizer_file_path) and os.path.exists(preprocessed_data_file_path):
        with open(tokenizer_file_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open(preprocessed_data_file_path, 'rb') as handle:
            data_dict = pickle.load(handle)
        return tokenizer, data_dict['X_train'], data_dict['X_test'], data_dict['y_train'], data_dict['y_test']
    else:
        data = pd.read_csv(data_path)
        data['text'] = data['text'].apply(clean_text).apply(trim_text)
        max_words = 10000
        max_len = 1000

        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(data['text'])
        sequences = tokenizer.texts_to_sequences(data['text'])
        data_pad = pad_sequences(sequences, maxlen=max_len)
        labels = np.where(data['source'] == 'Human', 1, 0)

        X_train, X_test, y_train, y_test = train_test_split(data_pad, labels, test_size=0.2, random_state=42)

        with open(tokenizer_file_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(preprocessed_data_file_path, 'wb') as handle:
            data_dict = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return tokenizer, X_train, X_test, y_train, y_test