from transformers import BertTokenizer, GPT2Tokenizer, DistilBertTokenizer, XLNetTokenizer, ElectraTokenizer
import torch


text = "Science is organized knowledge. Wisdom is organized life."
text = "The first and the best victory is to conquer self?"
text = "xcbxcbncn dfgryry tydhfg"
text = "“Despite the ostensibly insurmountable conundrum presented by the juxtaposition of quantum mechanics and general relativity, the relentless pursuit of a unified theory continues to galvanize the scientific community, engendering a plethora of hypotheses, each more intricate than the last.”"


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')


gpt2_encoded_text = gpt2_tokenizer.encode(text, return_tensors='pt')

bert_encoded_text = bert_tokenizer.encode(text, return_tensors='pt')
distilbert_encoded_text = distilbert_tokenizer.encode(text, return_tensors='pt')
electra_encoded_text = electra_tokenizer.encode(text, return_tensors='pt')


print(gpt2_encoded_text)
print("BERT Encoded Text: ", bert_encoded_text)
print("DistilBERT Encoded Text: ", distilbert_encoded_text)
print("ELECTRA Encoded Text: ", electra_encoded_text)
