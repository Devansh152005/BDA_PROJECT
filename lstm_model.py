import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("lstm_data.csv")

df = df.dropna()

# 🔥 convert safely (handles strings like "1", "1.0", etc.)
df["label"] = pd.to_numeric(df["label"], errors="coerce")

# remove invalid rows
df = df[df["label"].isin([0, 1])]

print("Shape after cleaning:", df.shape)
print("Labels:", df["label"].value_counts())

# 🔥 SAFE SAMPLING
if len(df) > 50000:
    df = df.sample(n=50000, random_state=42)



# -------------------------------
# Clean text
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df["text"] = df["text"].apply(clean_text)

# -------------------------------
# Tokenization
# -------------------------------
MAX_WORDS = 10000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])

X = tokenizer.texts_to_sequences(df["text"])
X = pad_sequences(X, maxlen=MAX_LEN)

y = df["label"].values

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model
# -------------------------------
model = Sequential([
    Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
    LSTM(64),
    Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# Training
# -------------------------------
model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# Save model + tokenizer
# -------------------------------
model.save("lstm_model.keras")

import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Model + Tokenizer saved")