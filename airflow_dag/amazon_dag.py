from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from utils.transformations import clean_review

spark = SparkSession.builder \
    .appName("amazon_pipeline") \
    .getOrCreate()

problem_ids = [
    "B088Z1YWBC","B0BLV1GNLN","B0BF57RN3K", ...
]

def step_bronze_to_silver():
    df = (
        spark.read.format("csv")
        .option("header", True)
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        .load("/Volumes/poc_amazon/bronze/src/amazon.csv")
    )

    df_problem = df.filter(col("product_id").isin(problem_ids))
    df_clean = df.filter(~col("product_id").isin(problem_ids))

    df_problem.write.mode("overwrite").option("header", True).csv("/Volumes/poc_amazon/bronze/src/problem_dataset/")
    df_clean.write.mode("overwrite").option("header", True).csv("/Volumes/poc_amazon/silver/clean_dataset/")

def step_silver_to_gold():
    df = spark.read.option("header", True).csv("/Volumes/poc_amazon/silver/clean_dataset/")

    clean_udf = udf(clean_review, StringType())
    df = df.withColumn("review_content", clean_udf(col("review_content")))

    df = df.withColumn(
        "sentiment",
        when(col("rating").isNull(), "unknown")
        .when((col("rating") >= 1.0) & (col("rating") < 3.0), "negative")
        .when((col("rating") >= 3.0) & (col("rating") < 4.0), "neutral")
        .when(col("rating") >= 4.0, "positive")
    )

    df_final = df.select("review_content","sentiment")

    df_final.write.mode("overwrite").csv("/Volumes/poc_amazon/gold/training_dataset/")

with DAG(
    dag_id="amazon_cleaning_pipeline",
    start_date=datetime(2023,1,1),
    schedule_interval=None,
    catchup=False
) as dag:

    t1 = PythonOperator(
        task_id="bronze_to_silver",
        python_callable=step_bronze_to_silver
    )

    t2 = PythonOperator(
        task_id="silver_to_gold",
        python_callable=step_silver_to_gold
    )

    t1 >> t2
