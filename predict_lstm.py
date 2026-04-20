import pickle
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model + tokenizer
model = load_model("lstm_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def predict(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    pred = model.predict(padded)[0][0]
    print("Score:", pred)

    if pred > 0.5:
        return "Positive 😊"
    else:
        return "Negative 😞"

# loop
while True:
    txt = input("\nEnter review: ")
    if txt == "exit":
        break
    print("Prediction:", predict(txt))