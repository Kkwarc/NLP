import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
# Ładowanie modelu ELMo
elmo = Elmo(options_file="path/to/elmo/options.json",
            weight_file="path/to/elmo/weights.hdf5",
            num_output_representations=1,
            dropout=0)
# Przykładowe zdania
sentences = ["Cześć, jak masz na imię?", "To jest przykładowe zdanie."]

# Konwersja zdań na indeksy znaków
character_ids = batch_to_ids(sentences)
# Uzyskiwanie reprezentacji ELMo
elmo_representations = elmo(character_ids)["elmo_representations"][0]
