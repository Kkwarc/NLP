import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from get_embedded_data import MAPPING
import ast


df = pd.read_csv("full_data.csv")


def reverse_dict(input_dict):
    reversed_dict = {}
    for key, value in input_dict.items():
        reversed_dict[value] = key
    return reversed_dict


def string_to_list(input_string):
    try:
        result_list = ast.literal_eval(input_string)
        if isinstance(result_list, list):
            return result_list
        else:
            raise ValueError("The input string does not represent a list.")
    except (ValueError, SyntaxError) as e:
        print(f"Error: {e}")
        return None


mapping = reverse_dict(MAPPING)
for i, row in df.iterrows():
    all_labels = [mapping[x] for x in string_to_list(row.all_labels)]
    all_predictions = [mapping[x] for x in string_to_list(row.all_predictions)]

    cm = confusion_matrix(all_labels, all_predictions, labels=list(MAPPING.keys()))

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(MAPPING.keys()), yticklabels=list(MAPPING.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f"confusion_matrix_{row.name}_{row.labels_to_delete}_{row.weigths}.png", dpi=500)
    plt.close()
