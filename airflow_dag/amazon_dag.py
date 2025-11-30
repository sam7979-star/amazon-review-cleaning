from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, length, trim, lower, udf, count
from pyspark.sql.types import StringType
from utils.transformations import clean_review

spark = SparkSession.builder.appName("amazon_pipeline").getOrCreate()

problem_ids = [
    "B088Z1YWBC","B0BLV1GNLN","B0BF57RN3K","B0B5LVS732","B09V12K8NT",
    "B09NVPSCQT","B0BF54972T","B0BF563HB4","B0BF4YBLPX","B0B3N7LR6K",
    "B09ZQK9X8G","B0BF54LXW6","B09PNKXSKF","B07WDKLRM4","B0BP18W8TM",
    "B07WHQWXL7","B09FKDH6FS","B07WGPKTS4","B097R25DP7","B09V17S2BG",
    "B0BNV7JM5Y","B0B53QFZPY","B07WGPKMP5","B09PLFJ7ZW","B0B53NXFFR",
    "B0949SBKMP","B0B53QLB9H","B0BMVWKZ8G","B0B5GF6DQD","B09YV463SW",
    "B0BNVBJW2S","B09BNXQ6BR","B0B2X35B1K","B09YV42QHZ","B09NVPJ3P4",
    "B0B3NDPCS9","B0BNXFDTZ2","B0B2RBP83P","B08S74GTBT"
]

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
    StructField("product_link", StringType(), True),
])
def bronze_to_silver():
    df = (
        spark.read.format("csv")
        .option("header", True)
        .schema(schema)
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        .load("/Volumes/poc_amazon/bronze/src/amazon.csv")
    )

    df_problem = df.filter(col("product_id").isin(problem_ids))
    df_clean = df.filter(~col("product_id").isin(problem_ids))

    df_problem.write.mode("overwrite").option("header", True).csv("/Volumes/poc_amazon/bronze/problem_rows/")
    df_clean.write.mode("overwrite").option("header", True).csv("/Volumes/poc_amazon/silver/clean_dataset/")

def silver_to_gold():
    df = spark.read.option("header", True).csv("/Volumes/poc_amazon/silver/clean_dataset/")

    clean_udf = udf(clean_review, StringType())
    df = df.withColumn("review_content", clean_udf(col("review_content")))

    df = df.filter(
        col("review_content").isNotNull() &
        (length(trim(col("review_content"))) > 0) &
        (~lower(trim(col("review_content"))).isin("nan", "none", "null", "n/a"))
    )

    # Drop duplicates (Silver layer rule)
    df = df.dropDuplicates(["user_id", "product_id", "review_content", "rating"])

    # Sentiment column
    df = df.withColumn(
        "sentiment",
        when(col("rating").isNull(), "unknown")
        .when((col("rating") >= 1.0) & (col("rating") < 3.0), "negative")
        .when((col("rating") >= 3.0) & (col("rating") < 4.0), "neutral")
        .when(col("rating") >= 4.0, "positive")
    )

    df = df.withColumnRenamed("review_content", "review_body")
    df_final = df.select("review_body", "sentiment")

    # Drop duplicate reviews again (Gold Layer rule)
    df_final = df_final.dropDuplicates(["review_body"])

    df_final.write.mode("overwrite").option("header", True).csv("/Volumes/poc_amazon/gold/training_dataset/")

with DAG(
    dag_id="amazon_cleaning_pipeline",
    start_date=datetime(2025,1,1),
    schedule_interval=None,
    catchup=False
) as dag:

    t1 = PythonOperator(task_id="bronze_to_silver", python_callable=bronze_to_silver)
    t2 = PythonOperator(task_id="silver_to_gold", python_callable=silver_to_gold)

    t1 >> t2
