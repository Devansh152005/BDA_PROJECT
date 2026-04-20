import numpy as np
from tensorflow.keras.models import load_model
import re

model = load_model("sentiment_model.h5")

VECTOR_SIZE = 1000

# -----------------------------------------
# Improved preprocessing (closer to Spark)
# -----------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()

    # remove simple stopwords (IMPORTANT)
    stopwords = set([
        "the","is","and","a","an","in","on","at","to","for","of","this","that","it"
    ])
    
    words = [w for w in words if w not in stopwords]

    return words

# -----------------------------------------
# Stable hashing (IMPORTANT FIX 🔥)
# -----------------------------------------
def text_to_vector(words):
    vector = np.zeros(VECTOR_SIZE)

    for word in words:
        index = abs(hash(word)) % VECTOR_SIZE   # 🔥 FIX: abs()
        vector[index] += 1

    return vector

# -----------------------------------------
# Prediction
# -----------------------------------------
def predict_sentiment(text):
    words = preprocess_text(text)
    vector = text_to_vector(words)

    prediction = model.predict(vector.reshape(1, -1))[0][0]

    print("Raw score:", prediction)

    if prediction > 0.5:
        return "Positive 😊"
    else:
        return "Negative 😞"

# -----------------------------------------
# Loop
# -----------------------------------------
while True:
    text = input("\nEnter review: ")
    
    if text.lower() == "exit":
        break
    
    print("Prediction:", predict_sentiment(text))