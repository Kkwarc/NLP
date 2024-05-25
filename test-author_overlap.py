import os
from collections import Counter
import itertools
import re


def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
    text = re.sub(r'[^a-zA-Z0-9\s,.!?]', '', text)
    return text


def get_top_words(text, X):
    words = text.split()
    counter = Counter(words)
    most_common_words = counter.most_common(X)
    return most_common_words


def calculate_overlap(words1, words2, X):
    set1 = set(word for word, count in words1)
    set2 = set(word for word, count in words2)

    common_words = set1 & set2
    presence_overlap = len(common_words) / len(set1.union(set2))

    return presence_overlap


if __name__ == "__main__":
    text_files = os.listdir("data_sets/quotes")

    texts = [read_text_file("data_sets/quotes/" + file_path) for file_path in text_files]

    top_words_list = [get_top_words(text, 50) for text in texts]

    results = []
    for (i, words1), (j, words2) in itertools.combinations(enumerate(top_words_list), 2):
        presence_overlap = calculate_overlap(words1, words2, 50)
        results.append((i, j, presence_overlap))

    results.sort(key=lambda x: x[2], reverse=True)

    for i, j, presence_overlap in results:
        print(f"Files {text_files[i]} i {text_files[j]}:")
        print(f"Overlap: {presence_overlap:.2f}")
