import pandas as pd


mapping = {
    "Aristotle": "aristotle",
    "Arthur-Schopenhauer-Quotes": "schopenhauer",
    "Friedrich-Nietzsche": "nietzsche",
    "Hegel": "hegel",
    "Immanuel-Kant": "kant",
    "Jean-Paul-Sartre": "sartre",
    "Plato": "plato",
    "Sigmund-Freud": "freud",
    "Spinoza": "spinoza"
}

quote_dictionary = "quotes/"

df = None
for name in mapping.keys():
    file_name = quote_dictionary + name + ".txt"
    with open(file_name, "r") as file:
        data = file.read()
        data = data.split("\n")
        data = [quote[1:] if quote.startswith(" ") else quote for quote in data]
        data = [quote if quote.endswith(".") else quote + "." for quote in data]
        data = [[mapping[name], quote + "."] for quote in data]
        print(data)
        temp_df = pd.DataFrame(data, columns=["author", "quote"])
        df = pd.concat([df, temp_df], ignore_index=True) if df is not None else temp_df
df = df.sort_values("author")
df = df.reset_index(drop=True)
counts = df.groupby('author').size()
print(counts)
df.to_csv("data_set.csv")
