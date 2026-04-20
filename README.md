Amazon Reviews Sentiment Analysis (Big Data + Deep Learning)

Project Overview
This project implements an end-to-end Big Data pipeline for sentiment analysis on the Amazon Reviews dataset using Hadoop HDFS for distributed storage, Apache Spark (PySpark) for preprocessing, and Deep Learning (TensorFlow/Keras) for classification.

The system evolves from a TF-IDF based model to a context-aware LSTM model, enabling better understanding of phrases like “not good”.

Architecture

Amazon Reviews Dataset
→ HDFS Storage
→ Spark Preprocessing (Cleaning + Labeling + Sampling)
→ TF-IDF Features → ANN Model
→ Raw Text → LSTM Model
→ Evaluation & Prediction

Technologies Used

Hadoop (HDFS)
Apache Spark (PySpark)
TensorFlow / Keras
Python (Pandas, NumPy, Scikit-learn)
Linux (Ubuntu)

Dataset

Source: Amazon Reviews Dataset (HuggingFace)
Format: Parquet (stored in HDFS)
Fields used:
text → Review content
rating → Converted into sentiment label

Data Processing Pipeline

Data Ingestion
Load dataset from HDFS using Spark.
Preprocessing
Remove null values
Convert text to lowercase
Remove punctuation
Tokenization and stopword removal
Convert rating into sentiment:
rating >= 3 → Positive (1)
rating < 3 → Negative (0)
Feature Engineering

TF-IDF Model:

HashingTF + IDF
Used for ANN model

LSTM Model:

Uses raw cleaned text
Converted into sequences
Sampling Strategy
Balanced sampling applied
Prevents memory issues
Reduces bias

Models Implemented

TF-IDF + ANN
Uses vectorized features
Fast but no context understanding
LSTM Model

Architecture:
Embedding → LSTM → Dense → Output

Captures word order and context
Handles phrases like “not good” correctly

Training Details

Dataset size: ~50,000 samples
Sequence length: 100
Vocabulary size: 10,000
Epochs: 3–5
Batch size: 64
CPU-based training

Evaluation Metrics

Accuracy
Precision
Recall
F1-score
Confusion Matrix

Example Predictions

Input: This product is amazing
Output: Positive

Input: not good at all
Output: Negative

Input: I regret buying this item
Output: Negative

How to Run

Run Spark Preprocessing

spark-submit --master spark://master:7077 --executor-memory 2G --total-executor-cores 8 preprocessing.py

Download Processed Data

hdfs dfs -get /bda/lstm_data/* .
cat part-* > lstm_data.csv

Train LSTM Model

source ~/dl_env/bin/activate
python lstm_model.py

Run Prediction

python predict_lstm.py

Limitations

LSTM trained on sampled data only
No distributed deep learning
Limited handling of sarcasm and spelling errors

Future Improvements

Use BERT or Transformer models
Add spell correction
Deploy as API
Integrate real-time streaming

Key Learnings

Difference between TF-IDF and LSTM
Importance of preprocessing at scale
Handling class imbalance
Integration of Big Data with Deep Learning

Author
Kanhaiya Chhaparwal<br>
Anuj Sule<br>
Devansh Upadhyay<br>
Yuvraj Srivastva<br>

Final Note
This project demonstrates how Big Data systems and Deep Learning models can be combined to build scalable NLP applications.
