spam.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# load data
df = pd.read_csv("spam.csv")

# input and output
X = df["text"]
y = df["label"]

# convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = MultinomialNB()
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# test message
msg = ["Free money now"]
msg = vectorizer.transform(msg)

print("Prediction:", model.predict(msg))