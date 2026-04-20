from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline

# -----------------------------------------
# 1. Spark Session
# -----------------------------------------
spark = SparkSession.builder \
    .appName("Amazon Reviews Preprocessing") \
    .getOrCreate()

# -----------------------------------------
# 2. Load Data from HDFS
# -----------------------------------------
df = spark.read.parquet("hdfs:///bda/amazon_reviews/")

print("✅ Data Loaded")
df.printSchema()

# -----------------------------------------
# 3. Select Required Columns
# -----------------------------------------
df = df.select(
    col("text").alias("reviewText"),
    col("rating").alias("overall")
)

# -----------------------------------------
# 4. Remove Nulls
# -----------------------------------------
df = df.dropna()

# -----------------------------------------
# 5. Create Sentiment Label
# -----------------------------------------
df = df.withColumn(
    "label",
    when(col("overall") >= 3, 1).otherwise(0)
)

# -----------------------------------------
# 6. Text Cleaning
# -----------------------------------------
df = df.withColumn("reviewText", lower(col("reviewText")))
df = df.withColumn("reviewText", regexp_replace(col("reviewText"), "[^a-zA-Z\\s]", ""))

# -----------------------------------------
# 7. ML Pipeline (REDUCED SIZE 🔥)
# -----------------------------------------
tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")

remover = StopWordsRemover(
    inputCol="words",
    outputCol="filtered"
)

hashingTF = HashingTF(
    inputCol="filtered",
    outputCol="rawFeatures",
    numFeatures=1000   # 🔥 reduced from 10000
)

idf = IDF(
    inputCol="rawFeatures",
    outputCol="features"
)

pipeline = Pipeline(stages=[
    tokenizer,
    remover,
    hashingTF,
    idf
])

# ----------------------------------------
# 8. Fit + Transform
# -----------------------------------------
model = pipeline.fit(df)
processed_df = model.transform(df)

# -----------------------------------------
# 9. Balanced Sampling (CRITICAL FIX 🔥)
# -----------------------------------------
positive_df = processed_df.filter(col("label") == 1).sample(fraction=0.005, seed=42)
negative_df = processed_df.filter(col("label") == 0).sample(fraction=0.005, seed=42)

sample_df = positive_df.union(negative_df)

print("Positive count:", positive_df.count())
print("Negative count:", negative_df.count())
print("Total sample:", sample_df.count())


# -----------------------------------------
# 10. Convert Vector → String (TF-IDF path)
# -----------------------------------------
final_df = sample_df.select(
    col("features").cast("string"),
    col("label")
)

final_df.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("hdfs:///bda/output")

print("✅ TF-IDF data saved")


# -----------------------------------------
# 11. Save TEXT for LSTM (IMPORTANT FIX 🔥)
# -----------------------------------------
clean_df = sample_df.select(
    col("reviewText").alias("text"),
    col("label")
)

clean_df.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("hdfs:///bda/lstm_data")

print("✅ LSTM text data saved")


spark.stop()