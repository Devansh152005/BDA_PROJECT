import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------------------
# 1. Load Data
# -----------------------------------------
df = pd.read_csv("sampled_data.csv")

# 🚨 REMOVE HEADER DUPLICATES
df = df[df["label"] != "label"]

# Convert label to int
df["label"] = df["label"].astype(int)

# -----------------------------------------
# 2. Parse Features (SAFE)
# -----------------------------------------
VECTOR_SIZE = 1000

def parse_vector(s):
    try:
        values_part = s.split("[")[-1].split("]")[0]
        values = np.fromstring(values_part, sep=",")
        
        padded = np.zeros(VECTOR_SIZE)
        padded[:min(len(values), VECTOR_SIZE)] = values[:VECTOR_SIZE]
        
        return padded
    except:
        return None

df["features"] = df["features"].apply(parse_vector)

# Drop bad rows
df = df[df["features"].notnull()]

# 🚨 REDUCE SIZE (VERY IMPORTANT)
df = df.sample(n=50000, random_state=42)

# Convert to matrix
X = np.vstack(df["features"].values).astype(np.float32)
y = df["label"].values.astype(np.float32)

print("Shape:", X.shape)
print("Label distribution:\n", pd.Series(y).value_counts())

# -----------------------------------------
# 3. Balance Dataset (SAFE)
# -----------------------------------------
df_bal = pd.DataFrame(X)
df_bal["label"] = y

df_majority = df_bal[df_bal.label == 1]
df_minority = df_bal[df_bal.label == 0]

if len(df_minority) == 0:
    print("⚠ No minority class — skipping balancing")
    df_balanced = df_bal
else:
    df_minority_up = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )
    df_balanced = pd.concat([df_majority, df_minority_up])

X = df_balanced.drop("label", axis=1).values.astype(np.float32)
y = df_balanced["label"].values.astype(np.float32)

# -----------------------------------------
# 4. Train-Test Split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------
# 5. Model (LIGHT + STABLE)
# -----------------------------------------
model = Sequential([
    Dense(32, activation='relu', input_shape=(VECTOR_SIZE,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------------
# 6. Train
# -----------------------------------------
model.fit(
    X_train, y_train,
    epochs=60,              # reduced for stability
    batch_size=64
)

# -----------------------------------------
# 7. Evaluate
# -----------------------------------------
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


model.save("sentiment_model.h5")
print("✅ Model saved")