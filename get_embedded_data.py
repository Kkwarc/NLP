from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np
import torch
import torchtext
import torchtext.vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

mapping = {
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

def get_sentences():
    with open('data_set.csv', 'r', encoding='utf-8') as dh:
        list_of_words = []
        list_of_targets = []
        for i, line in enumerate(dh):
            if i > 0:
                line = line.strip()
                line = line.split('@')
                line[-1] = word_tokenize(line[-1].lower())
                list_of_words.append(line[-1])
                list_of_targets.append(mapping[line[1]])
        dh.close()
    return list_of_words, list_of_targets

def get_data_word2vec_CNN(batch):
    max_len_of_sentence = 125
    list_of_words, list_of_targets = get_sentences()
    model = Word2Vec.load('word2vec/word2vec_100d')
    word2vec_embeddings = []
    for sentence in list_of_words:
        emb_sentence = np.empty((100,0))
        for i in range(max_len_of_sentence):
            if i < len(sentence):
                emb_sentence = np.hstack((emb_sentence, np.reshape(model.wv[sentence[i]], (100, 1))))
            else:
                emb_sentence = np.hstack((emb_sentence, np.zeros((100,1))))
        word2vec_embeddings.append(torch.from_numpy(emb_sentence))
    dataset = sentence_dataset(word2vec_embeddings, list_of_targets)
    dataloader = DataLoader(dataset, batch, shuffle=True, drop_last=True)
    return dataloader

def get_data_word2vec_LSTM(batch):
    list_of_words, list_of_targets = get_sentences()
    model = Word2Vec.load('word2vec/word2vec_100d')
    word2vec_embeddings = []
    for sentence in list_of_words:
        emb_sentence = np.empty((100,0))
        for word in sentence:
            emb_sentence = np.hstack((emb_sentence, np.reshape(model.wv[word], (100, 1))))
        word2vec_embeddings.append(torch.from_numpy(emb_sentence).T)
    dataset = sentence_dataset(word2vec_embeddings, list_of_targets)
    dataloader = DataLoader(dataset, batch, shuffle=True, drop_last=True, collate_fn=pad_collate_fn)
    return dataloader

def get_data_glove_CNN(batch):
    max_len_of_sentence = 125
    list_of_words, list_of_targets = get_sentences()
    glove = torchtext.vocab.GloVe(name="6B", dim=100)   
    glove_embeddings = []
    for sentence in list_of_words:
        emb_sentence = torch.empty((100,0),dtype=torch.double)
        for i in range(max_len_of_sentence):
            if i < len(sentence):
                emb_sentence = torch.hstack((emb_sentence, torch.reshape(glove[sentence[i]], (100, 1))))
            else:
                emb_sentence = torch.hstack((emb_sentence, torch.zeros((100,1))))
        glove_embeddings.append(emb_sentence)
    dataset = sentence_dataset(glove_embeddings, list_of_targets)
    dataloader = DataLoader(dataset, batch, shuffle=True, drop_last=True)
    return dataloader

def get_data_glove_LSTM(batch):
    list_of_words, list_of_targets = get_sentences()
    glove = torchtext.vocab.GloVe(name="6B", dim=100)
    glove_embeddings = []
    for sentence in list_of_words:
        emb_sentence = torch.empty((100,0))
        for word in sentence:
            emb_sentence = torch.hstack((emb_sentence, torch.reshape(glove[word], (100, 1))))
        glove_embeddings.append(emb_sentence.T)
    dataset = sentence_dataset(glove_embeddings, list_of_targets)
    dataloader = DataLoader(dataset, batch, shuffle=True, drop_last=True, collate_fn=pad_collate_fn)
    return dataloader
