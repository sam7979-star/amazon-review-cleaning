from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, length, trim, lower, udf, count
from pyspark.sql.types import StringType, StructType, StructField, DoubleType
import shutil, os

# Import clean_review()
from utils.transformations import clean_review


# ------------------------------------------------------------------------------------
# Spark Session 
# ------------------------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("amazon_cleaning_pipeline") \
    .getOrCreate()


# ------------------------------------------------------------------------------------
# Schema 
# ------------------------------------------------------------------------------------
schema = StructType([
    StructField("product_id", StringType(), True),
    StructField("product_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("discounted_price", StringType(), True),
    StructField("actual_price", StringType(), True),
    StructField("discount_percentage", StringType(), True),
    StructField("rating", DoubleType(), True),
    StructField("rating_count", StringType(), True),
    StructField("about_product", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("user_name", StringType(), True),
    StructField("review_id", StringType(), True),
    StructField("review_title", StringType(), True),
    StructField("review_content", StringType(), True),
    StructField("img_link", StringType(), True),
    StructField("product_link", StringType(), True)
])


# ------------------------------------------------------------------------------------
# STEP 1 – BRONZE → SILVER
# Reads the raw Amazon CSV from Bronze and writes cleaned structure to Silver
# ------------------------------------------------------------------------------------
def bronze_to_silver():
    df = (
        spark.read.format("csv")
        .schema(schema)
        .option("header", True)
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        .load("/Volumes/poc_amazon/bronze/src/amazon.csv")
    )

    # Write directly to Silver
    df.write.mode("overwrite").option("header", True).csv("/Volumes/poc_amazon/silver/clean_dataset/")


# ------------------------------------------------------------------------------------
# STEP 2 – SILVER → GOLD
# ------------------------------------------------------------------------------------
def silver_to_gold():
    df = (
        spark.read.format("csv")
        .schema(schema)
        .option("header", True)
        .load("/Volumes/poc_amazon/silver/clean_dataset/")
    )

    # Register UDF
    clean_udf = udf(clean_review, StringType())

    # Apply cleaning on review_content
    df = df.withColumn("review_content", clean_udf(col("review_content")))

    # Sentiment column
    df = df.withColumn(
        "sentiment",
        when(col("rating").isNull(), "unknown")
        .when((col("rating") >= 1.0) & (col("rating") < 3.0), "negative")
        .when((col("rating") >= 3.0) & (col("rating") < 4.0), "neutral")
        .when(col("rating") >= 4.0, "positive")
    )

    # Data Quality Checks
    df = df.filter(
        col("review_content").isNotNull() &
        (length(trim(col("review_content"))) > 0) &
        (~lower(trim(col("review_content"))).isin("null", "none", "nan", "n/a"))
    )

    # Drop duplicates
    df = df.dropDuplicates(["user_id", "product_id", "review_content", "rating"])

    # Rename column
    df = df.withColumnRenamed("review_content", "review_body")

    # Final selection
    df_final = df.select("review_body", "sentiment")

    # Drop duplicates in final output
    df_final = df_final.dropDuplicates(["review_body"])

    # TEMP OUTPUT
    temp_path = "/Volumes/poc_amazon/gold/output/training_dataset_temp/"

    df_final.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_path)

    # Move & rename part file to training_dataset.csv
    src_folder = temp_path.replace("dbfs:", "/dbfs")
    target_folder = "/dbfs/Volumes/poc_amazon/gold/output/training_dataset/"
    target_filename = "training_dataset.csv"

    os.makedirs(target_folder, exist_ok=True)

    for file in os.listdir(src_folder):
        if file.startswith("part-") and file.endswith(".csv"):
            shutil.move(
                os.path.join(src_folder, file),
                os.path.join(target_folder, target_filename)
            )
            break


# ------------------------------------------------------------------------------------
# DAG Definition
# ------------------------------------------------------------------------------------
with DAG(
    dag_id="amazon_cleaning_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args={"owner": "sam"}
) as dag:

    t1 = PythonOperator(
        task_id="bronze_to_silver",
        python_callable=bronze_to_silver
    )

    t2 = PythonOperator(
        task_id="silver_to_gold",
        python_callable=silver_to_gold
    )

    t1 >> t2
