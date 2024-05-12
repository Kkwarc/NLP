from transformers import BertModel, ElectraModel, RobertaModel
from transformers import BertTokenizer, ElectraTokenizer, RobertaTokenizer


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# robert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# robert_model = RobertaModel.from_pretrained('roberta-base')
#
# electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
# electra_model = ElectraModel.from_pretrained('google/electra-small-discriminator')


text = "The Life of the intellect is the best and pleasantest for man, because the intellect more than anything else is the man. Thus it will be the happiest life as well."

# encoded_input = bert_tokenizer(text)
# print(encoded_input)
encoded_input = bert_tokenizer(text, return_tensors='pt', add_special_tokens=False)
output = bert_model(**encoded_input)
text_embedding = output.last_hidden_state[0]
print(output)
print(text_embedding)
# print(middle_tensor)


# encoded_input = robert_tokenizer(text)
# print(encoded_input)
# encoded_input = robert_tokenizer(text, return_tensors='pt')
# output = robert_model(**encoded_input)
# print(output)
# text_embedding = output.last_hidden_state[0]
# print(text_embedding)
#
#
# encoded_input = electra_tokenizer(text)
# print(encoded_input)
# encoded_input = electra_tokenizer(text, return_tensors='pt')
# output = electra_model(**encoded_input)
# print(output)
# text_embedding = output.last_hidden_state[0]
# print(text_embedding)
