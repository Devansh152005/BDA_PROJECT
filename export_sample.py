from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Export Sample").getOrCreate()

# Load processed data
df = spark.read.parquet("hdfs:///bda/processed/train")

# Take small sample (important for DL)
sample_df = df.sample(fraction=0.01, seed=42)

# Convert to Pandas
pdf = sample_df.toPandas()

# Save locally
pdf.to_csv("train_sample.csv", index=False)

print("✅ Sample exported to train_sample.csv")