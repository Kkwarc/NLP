import pandas as pd


df = pd.read_csv("data_set.csv", sep="@")


for author in df["author"].to_list():
    temp_df = df[df["author"] == author]
    result = '\n'.join(temp_df["quote"].str.lower())

    with open(f'author_texts_and_word_clouds/{author}.txt', "w") as file:
        file.write(result)
