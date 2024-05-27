from transformers import BertModel, ElectraModel, RobertaModel
from transformers import BertTokenizer, ElectraTokenizer, RobertaTokenizer
import torch
import torch.nn.functional as F
from get_embedded_data import MAPPING
import torchtext
import numpy as np
import torchtext.vocab
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


CLASS_LABELS = MAPPING.keys()

MODELS_NAME_MAPPING = {
    "Bert_full_weights": ("Bert", "bert_[]_[1, 1, 2, 3, 1.33, 1.33, 5, 1, 2.5].pt"),
    "Bert_full_no_weights": ("Bert", "bert_[]_[].pt"),
    "Bert_balanced_weights": ("Bert", "bert_['spinoza', 'hegel', 'plato']_[].pt"),
    "Bert_balanced_no_weights": ("Bert", "bert_['spinoza', 'hegel', 'plato']_[1, 1, 2, 3, 1.33, 1.33, 5, 1, 2.5].pt"),
    "Roberta_full_weights": ("Roberta", "roberta_[]_[1, 1, 2, 3, 1.33, 1.33, 5, 1, 2.5].pt"),
    "Roberta_full_no_weights": ("Roberta", "roberta_[]_[].pt"),
    "Roberta_balanced_weights": ("Roberta", "roberta_['spinoza', 'hegel', 'plato']_[1, 1, 2, 3, 1.33, 1.33, 5, 1, 2.5].pt"),
    "Roberta_balanced_no_weights": ("Roberta", "roberta_['spinoza', 'hegel', 'plato']_[].pt"),
    "Electra_full_weights": ("Electra", "electra_[]_[1, 1, 2, 3, 1.33, 1.33, 5, 1, 2.5].pt"),
    "Electra_full_no_weights": ("Electra", "electra_[]_[].pt"),
    "Electra_balanced_weights": ("Electra", "electra_['spinoza', 'hegel', 'plato']_[1, 1, 2, 3, 1.33, 1.33, 5, 1, 2.5].pt"),
    "Electra_balanced_no_weights": ("Electra", "electra_['spinoza', 'hegel', 'plato']_[].pt"),
    "CNN_glove_full_weights": ("CNN", "glove", "CNN_glove_[]_[1, 1, 2, 3, 1.33, 1.33, 5, 1, 2.5]"),
    "CNN_glove_full_no_weights": ("CNN", "glove", "CNN_glove_[]_[]"),
    "CNN_glove_balanced_weights": ("CNN", "glove", "CNN_glove_['spinoza', 'hegel', 'plato']_[]"),
    "CNN_glove_balanced_no_weight": ("CNN", "glove", "CNN_glove_['spinoza', 'hegel', 'plato']_[1, 1, 2, 3, 1.33, 1.33, 5, 1, 2.5]"),
    "CNN_word2vec_full_weights": ("CNN", "word2vec", "CNN_word2vec_[]_[1, 1, 2, 3, 1.33, 1.33, 5, 1, 2.5]"),
    "CNN_word2vec_full_no_weights": ("CNN", "word2vec", "CNN_word2vec_[]_[]"),
    "CNN_word2vec_balanced_weights": ("CNN", "word2vec", "CNN_word2vec_['spinoza', 'hegel', 'plato']_[]"),
    "CNN_word2vec_balanced_no_weight": ("CNN", "word2vec", "CNN_word2vec_['spinoza', 'hegel', 'plato']_[1, 1, 2, 3, 1.33, 1.33, 5, 1, 2.5]"),
    "LSTM_glove_full_weights": ("LSTM", "glove", "LSTM_glove_[]_[1, 1, 2, 3, 1.33, 1.33, 5, 1, 2.5]"),
    "LSTM_glove_full_no_weights": ("LSTM", "glove", "LSTM_glove_[]_[]"),
    "LSTM_glove_balanced_weights": ("LSTM", "glove", "LSTM_glove_['spinoza', 'hegel', 'plato']_[]"),
    "LSTM_glove_balanced_no_weight": ("LSTM", "glove", "LSTM_glove_['spinoza', 'hegel', 'plato']_[1, 1, 2, 3, 1.33, 1.33, 5, 1, 2.5]"),
    "LSTM_word2vec_full_weights": ("LSTM", "word2vec", "LSTM_word2vec_[]_[1, 1, 2, 3, 1.33, 1.33, 5, 1, 2.5]"),
    "LSTM_word2vec_full_no_weights": ("LSTM", "word2vec", "LSTM_word2vec_[]_[]"),
    "LSTM_word2vec_balanced_weights": ("LSTM", "word2vec", "LSTM_word2vec_['spinoza', 'hegel', 'plato']_[]"),
    "LSTM_word2vec_balanced_no_weight": ("LSTM", "word2vec", "LSTM_word2vec_['spinoza', 'hegel', 'plato']_[1, 1, 2, 3, 1.33, 1.33, 5, 1, 2.5]"),
}


def load_model_tokenizer(chosen_model_name):
    nn = torch.load(f"train_models/{chosen_model_name[1]}")
    match chosen_model_name[0]:
        case "Bert":
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
            return model, tokenizer, nn
        case "Roberta":
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            model = RobertaModel.from_pretrained('roberta-base')
            return model, tokenizer, nn
        case "Electra":
            tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
            model = ElectraModel.from_pretrained('google/electra-small-discriminator')
            return model, tokenizer, nn


def encode_quote(sentence, model, tokenizer, device, max_length=125):
    model = model.to(device)
    encoded_input = tokenizer(sentence, return_tensors='pt', add_special_tokens=False, pad_to_max_length=True,
                              max_length=max_length)
    encoded_input = encoded_input.to(device)
    with torch.no_grad():
        output = model(**encoded_input)
    return output.last_hidden_state[0]


def predict(encoded_output, network):
    input = torch.flatten(encoded_output.double(), 0)
    outputs = network(input)
    return outputs


def get_prediction_probabilities(outputs):
    probabilities = F.softmax(outputs, dim=0)
    return probabilities


def prepare_glove(quote, model, max_len_of_sentence, device):
    glove = torchtext.vocab.GloVe(name="6B", dim=100)
    emb_sentence = torch.empty((100,0),dtype=torch.double)
    if model == "LSTM":
        for word in quote:
            emb_sentence = torch.hstack((emb_sentence, torch.reshape(glove[word], (100, 1))))
        return emb_sentence.T.unsqueeze(0).to(device)
    else:
        for i in range(max_len_of_sentence):
            if i < len(quote):
                emb_sentence = torch.hstack((emb_sentence, torch.reshape(glove[quote[i]], (100, 1))))
            else:
                emb_sentence = torch.hstack((emb_sentence, torch.zeros((100,1))))
        return emb_sentence.unsqueeze(0).to(device)


def prepare_word2vec(quote, model, max_len_of_sentence, device):
    tok = Word2Vec.load('word2vec/word2vec_100d')
    emb_sentence = np.empty((100,0))
    if model == "LSTM":
        for word in quote:
            emb_sentence = np.hstack((emb_sentence, np.reshape(tok.wv[word], (100, 1))))
        return torch.from_numpy(emb_sentence).T.unsqueeze(0).to(device)
    else:
        for i in range(max_len_of_sentence):
            if i < len(quote):
                emb_sentence = np.hstack((emb_sentence, np.reshape(tok.wv[quote[i]], (100, 1))))
            else:
                emb_sentence = np.hstack((emb_sentence, np.zeros((100,1))))
        return torch.from_numpy(emb_sentence).unsqueeze(0).to(device)


def main(chosen_model, device, quote):
    max_len_of_sentence = 125
    chosen_model = MODELS_NAME_MAPPING[chosen_model]
    device = torch.device(device)
    if chosen_model[0] == "Bert" or chosen_model[0] == "Roberta" or chosen_model[0] == "Electra":
        model, tokenizer, nn = load_model_tokenizer(chosen_model)
        encoded_output = encode_quote(quote, model, tokenizer, device, max_length=max_len_of_sentence)
        output = predict(encoded_output, nn)
    else:
        quote = word_tokenize(quote.lower())
        if chosen_model[1] == "glove":
            endoded_input = prepare_glove(quote, chosen_model[0], max_len_of_sentence, device)
        else:
            endoded_input = prepare_word2vec(quote, chosen_model[0], max_len_of_sentence, device)
        model = torch.load(f"train_models/{chosen_model[2]}")
        if chosen_model[0] == "LSTM":
            output = model(endoded_input)
            output = output[0][len(quote) - 1]
        else:
            output = model(endoded_input)
            output = output[0]
    probabilities = get_prediction_probabilities(output)

    for label, probability in zip(CLASS_LABELS, probabilities):
        print(f'Class: {label}, Probability: {probability.item():.2f}')
    return probabilities


"""
Models names:
Bert_full_weights
Bert_full_no_weights
Bert_balanced_weights"
Bert_balanced_no_weights
Roberta_full_weights
Roberta_full_no_weights
Roberta_balanced_weights
Roberta_balanced_no_weights
Electra_full_weights
Electra_full_no_weights
Electra_balanced_weights
Electra_balanced_no_weights
CNN_glove_full_weights
CNN_glove_full_no_weights
CNN_glove_balanced_weights
CNN_glove_balanced_no_weight
CNN_word2vec_full_weights
CNN_word2vec_full_no_weights
CNN_word2vec_balanced_weights
CNN_word2vec_balanced_no_weight
LSTM_glove_full_weights
LSTM_glove_full_no_weights
LSTM_glove_balanced_weights
LSTM_glove_balanced_no_weight
LSTM_word2vec_full_weights
LSTM_word2vec_full_no_weights
LSTM_word2vec_balanced_weights
LSTM_word2vec_balanced_no_weight
"""


if __name__ == "__main__":
    chosen_model = "LSTM_word2vec_full_no_weights"  # Here is the place to copy chosen model name
    device = "cuda"  # cuda or cpu
    quote = "The men is wise"  # Here is the place for your quote

    main(chosen_model, device, quote)
