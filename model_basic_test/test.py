from transformers import BertModel, ElectraModel, RobertaModel
from transformers import BertTokenizer, ElectraTokenizer, RobertaTokenizer


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

robert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
robert_model = RobertaModel.from_pretrained('roberta-base')

electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
electra_model = ElectraModel.from_pretrained('google/electra-small-discriminator')


text = "This"

encoded_input = bert_tokenizer(text)
print(encoded_input)
encoded_input = bert_tokenizer(text, return_tensors='pt')
output = bert_model(**encoded_input)
print(output.ndim)
text_embedding = output.last_hidden_state[0]
print(text_embedding)


encoded_input = robert_tokenizer(text)
print(encoded_input)
encoded_input = robert_tokenizer(text, return_tensors='pt')
output = robert_model(**encoded_input)
print(output.ndim)
text_embedding = output.last_hidden_state[0]
print(text_embedding)


encoded_input = electra_tokenizer(text)
print(encoded_input)
encoded_input = electra_tokenizer(text, return_tensors='pt')
output = electra_model(**encoded_input)
print(output.ndim)
text_embedding = output.last_hidden_state[0]
print(text_embedding)
