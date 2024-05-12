import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv("data_set.csv", sep="@")

# Transform labels
def transform_labels(row):
    x = {"aristotle": 0,
         "schopenhauer": 1,
         "nietzsche": 2,
         "hegel": 3,
         "kant": 4,
         "sartre": 5,
         "plato": 6,
         "freud": 7,
         "spinoza": 8}
    return x[row]

df["label"] = df["author"].apply(transform_labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df["quote"], df["label"], test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Initialize MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=100, activation='relu', solver='adam', random_state=42)

# Train the MLPClassifier
mlp.fit(X_train, y_train)

# Evaluate the MLPClassifier
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
