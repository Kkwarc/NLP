from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from transformers import BertTokenizer
import numpy as np
import pandas as pd
import torch
import torchtext
import torchtext.vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


MAPPING = {
    "aristotle" : 0,
    "schopenhauer": 1,
    "nietzsche": 2,
    "hegel": 3,
    "kant": 4,
    "sartre": 5,
    "plato": 6,
    "freud": 7,
    "spinoza": 8
}


def pad_collate_fn(batch, pad_value=0):
    xx, yy = zip(*batch)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_value)
    x_lens = [len(x) for x in xx]
    return torch.tensor(xx_pad, dtype=torch.double), torch.tensor(yy), x_lens

class sentence_dataset(torch.utils.data.Dataset):
    def __init__(self, sentence, target):
        self.data = list(zip(sentence,target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        in_data, target = self.data[idx]
        return in_data, target


def split_data(df, labels_column_name, values_column_name, test_size=0.2, separator=None, mapping=None, labels_to_delete=[]):
    def map_labels(value):
        return mapping[value]
    if type(df) == str:
        df = pd.read_csv(df, sep=separator)
    unique_labels = df[labels_column_name].unique()
    print(unique_labels)
    unique_labels_list = [df[df[labels_column_name] == label]  for label in unique_labels if label not in labels_to_delete]

    X_train, X_test, y_train, y_test = [], [], [], []

    for df in unique_labels_list:
        if mapping is not None:
            df["labels"] = df[labels_column_name].apply(map_labels)
        temp_X_train, temp_X_test, temp_y_train, temp_y_test = train_test_split(df[values_column_name].to_list(), df["labels"].to_list(), test_size=test_size)
        X_train += temp_X_train
        X_test += temp_X_test
        y_train += temp_y_train
        y_test += temp_y_test

    return X_train, X_test, y_train, y_test


# def get_sentences_transformers():
#     with open('data_set.csv', 'r', encoding='utf-8') as dh:
#         list_of_words = []
#         list_of_targets = []
#         for i, line in enumerate(dh):
#             if i > 0:
#                 line = line.strip()
#                 line = line.split('@')
#                 list_of_words.append(line[-1].lower())
#                 list_of_targets.append(MAPPING[line[1]])
#         dh.close()
#     return list_of_words, list_of_targets


# def get_sentences():
#     with open('data_set.csv', 'r', encoding='utf-8') as dh:
#         list_of_words = []
#         list_of_targets = []
#         for i, line in enumerate(dh):
#             if i > 0:
#                 line = line.strip()
#                 line = line.split('@')
#                 line[-1] = word_tokenize(line[-1].lower())
#                 list_of_words.append(line[-1])
#                 list_of_targets.append(MAPPING[line[1]])
#         dh.close()
#     return list_of_words, list_of_targets


def get_data_word2vec_CNN(batch, words:list, labels:list,):
    max_len_of_sentence = 125
    model = Word2Vec.load('word2vec/word2vec_100d')
    word2vec_embeddings = []
    for sentence in words:
        sentence = word_tokenize(sentence)
        emb_sentence = np.empty((100,0))
        for i in range(max_len_of_sentence):
            if i < len(sentence):
                emb_sentence = np.hstack((emb_sentence, np.reshape(model.wv[sentence[i]], (100, 1))))
            else:
                emb_sentence = np.hstack((emb_sentence, np.zeros((100,1))))
        word2vec_embeddings.append(torch.from_numpy(emb_sentence))
    dataset = sentence_dataset(word2vec_embeddings, labels)
    dataloader = DataLoader(dataset, batch, shuffle=True, drop_last=True)
    return dataloader


def get_data_word2vec_LSTM(batch, words:list, labels:list,):
    model = Word2Vec.load('word2vec/word2vec_100d')
    word2vec_embeddings = []
    for sentence in words:
        sentence = word_tokenize(sentence)
        emb_sentence = np.empty((100,0))
        for word in sentence:
            emb_sentence = np.hstack((emb_sentence, np.reshape(model.wv[word], (100, 1))))
        word2vec_embeddings.append(torch.from_numpy(emb_sentence).T)
    dataset = sentence_dataset(word2vec_embeddings, labels)
    dataloader = DataLoader(dataset, batch, shuffle=True, drop_last=True, collate_fn=pad_collate_fn)
    return dataloader


def get_data_glove_CNN(batch, words:list, labels:list,):
    max_len_of_sentence = 125
    glove = torchtext.vocab.GloVe(name="6B", dim=100)
    glove_embeddings = []
    for sentence in words:
        sentence = word_tokenize(sentence)
        emb_sentence = torch.empty((100,0),dtype=torch.double)
        for i in range(max_len_of_sentence):
            if i < len(sentence):
                emb_sentence = torch.hstack((emb_sentence, torch.reshape(glove[sentence[i]], (100, 1))))
            else:
                emb_sentence = torch.hstack((emb_sentence, torch.zeros((100,1))))
        glove_embeddings.append(emb_sentence)
    dataset = sentence_dataset(glove_embeddings, labels)
    dataloader = DataLoader(dataset, batch, shuffle=True, drop_last=True)
    return dataloader


def get_data_glove_LSTM(batch, words:list, labels:list,):
    glove = torchtext.vocab.GloVe(name="6B", dim=100)
    glove_embeddings = []
    for sentence in words:
        sentence = word_tokenize(sentence)
        emb_sentence = torch.empty((100,0))
        for word in sentence:
            emb_sentence = torch.hstack((emb_sentence, torch.reshape(glove[word], (100, 1))))
        glove_embeddings.append(emb_sentence.T)
    dataset = sentence_dataset(glove_embeddings, labels)
    dataloader = DataLoader(dataset, batch, shuffle=True, drop_last=True, collate_fn=pad_collate_fn)
    return dataloader


def get_data_tokenizer_MLP(batch, words:list, labels:list, device, tokenizer, model):
    model = model.to(device)
    max_len_of_sentence = 125
    bert_embeddings = []
    for sentence in words:
        encoded_input = tokenizer(sentence, return_tensors='pt', add_special_tokens=False, pad_to_max_length=True,
                                       max_length=max_len_of_sentence)
        encoded_input = encoded_input.to(device)
        with torch.no_grad():
            output = model(**encoded_input)
        text_embedding = output.last_hidden_state[0]
        bert_embeddings.append(text_embedding)

    shape = text_embedding.shape[1]
    dataset = sentence_dataset(bert_embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True)
    return dataloader, shape
