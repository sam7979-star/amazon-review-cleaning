# Databricks notebook source
# MAGIC %md
# MAGIC ##Step:1 Loading the file into the spark for processing##

# COMMAND ----------

raw_data = spark.read.option("header", True).csv("/Volumes/poc_amazon/bronze/src/amazon.csv")
raw_data.printSchema()



# COMMAND ----------

raw_data.coalesce(1).write.mode("overwrite").option("header", True).csv("/Volumes/poc_amazon/bronze/src/New_raw_dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC ###File Having formatting issues###

# COMMAND ----------

# MAGIC %md
# MAGIC ###59 records are corrupt moved to different location###

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import col, regexp_replace, trim, lit, length, when, count

# COMMAND ----------

# MAGIC %md
# MAGIC ####Problem ID's####

# COMMAND ----------

problem_ids = [
"B088Z1YWBC","B0BLV1GNLN","B0BF57RN3K","B0B5LVS732","B09V12K8NT",
"B09NVPSCQT","B0BF54972T","B0BF563HB4","B0BF4YBLPX","B0B3N7LR6K",
"B09ZQK9X8G","B0BF54LXW6","B09PNKXSKF","B07WDKLRM4","B0BP18W8TM",
"B07WHQWXL7","B09FKDH6FS","B07WGPKTS4","B097R25DP7","B09V17S2BG",
"B0BNV7JM5Y","B0B53QFZPY","B07WGPKMP5","B09PLFJ7ZW","B0B53NXFFR",
"B0949SBKMP","B0B53QLB9H","B0BMVWKZ8G","B0B5GF6DQD","B09YV463SW",
"B0BNVBJW2S","B09BNXQ6BR","B0B2X35B1K","B09YV42QHZ","B09NVPJ3P4",
"B0B3NDPCS9","B0BNXFDTZ2","B0B5LVS732","B09V12K8NT","B09NVPSCQT",
"B0B3N7LR6K","B09ZQK9X8G","B08MCD9JFY","B097R25DP7","B09P18XVW6",
"B07Z1X6VFC","B0949SBKMP","B07Z1YVP72","B08JMC1988","B00N1U7JXM",
"B09RKFBCV7","B07Z53L5QL","B09NC2TY11","B08KRMK9LZ","B094JB13XL",
"B07Z1Z77ZZ","B09939XJX8","B0B2RBP83P","B08S74GTBT"
]

# COMMAND ----------

df = (
    spark.read.format("csv")
    .option("header", True)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')
    .option("ignoreLeadingWhiteSpace", True)
    .option("ignoreTrailingWhiteSpace", True)
    .option("mode", "PERMISSIVE")
    .load("/Volumes/poc_amazon/bronze/src/amazon.csv")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ###Filtering the Problem Records###

# COMMAND ----------

df_problem = df.filter(df.product_id.isin(problem_ids))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Filtering the correct records###

# COMMAND ----------

df_clean = df.filter(~df.product_id.isin(problem_ids))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Writing the Corrupt Records into different SRC###

# COMMAND ----------

df_problem.write.mode("overwrite").option("header", True).csv("/Volumes/poc_amazon/bronze/src/problem_dataset/")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Storing Clean Record into Source###

# COMMAND ----------

df_clean.write.mode("overwrite").option("header", True).csv("/Volumes/poc_amazon/bronze/src/clean_dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Giving Schema to each and every column for clean records###

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ###Reading the Clean Dataset###

# COMMAND ----------

data = "/Volumes/poc_amazon/bronze/src/clean_dataset/part-00000-tid-4601255286852897246-37ff8c9a-5596-444b-b88c-cbf4cee0a685-132-1-c000.csv"
df = spark.read.format("csv").schema(schema).option("header", True).load(data)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #Step:2 Cleaning the records#

# COMMAND ----------

# MAGIC %md
# MAGIC ###Reading Transformations.py###

# COMMAND ----------

import os
import sys
print(os.path.join(os.getcwd(),'..','..'))


# COMMAND ----------

project_path = os.path.join(os.getcwd(),'..','..')
sys.path.append(project_path)
from utils.transformations import clean_review


# COMMAND ----------

# MAGIC %md
# MAGIC ###Applying Transformation.py to clean Dataframe review_content###

# COMMAND ----------

from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# Register clean_review as a UDF
clean_udf = udf(
    clean_review,
    StringType()
)

df = df.withColumn(
    "review_content",
    clean_udf(col("review_content"))
)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Step:3 Creating Sentiment column based on Rating##

# COMMAND ----------

# MAGIC %md
# MAGIC ##Creating a new Column Sentiment for Rating##

# COMMAND ----------

from pyspark.sql.functions import col, when

df = df.withColumn(
    "sentiment",
    when(col("rating").isNull(), "unknown")
    .when((col("rating") >= 1.0) & (col("rating") < 3.0), "negative")
    .when((col("rating") >= 3.0) & (col("rating") < 4.0), "neutral")
    .when(col("rating") >= 4.0, "positive")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Step:4 Handling the Data Quality ##

# COMMAND ----------

# MAGIC %md
# MAGIC #Handle Data Quality Issues#

# COMMAND ----------

# MAGIC %md
# MAGIC ####Missing OR empty review_content####

# COMMAND ----------

from pyspark.sql.functions import length, trim, lower
df = df.filter(col("review_content").isNotNull() & (length(trim(col("review_content")))>0)&(~lower(trim(col("review_content"))).isin("null","none","nan","n/a")))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Dropping Duplicate based on 4 columns###

# COMMAND ----------

df = df.dropDuplicates(["user_id","product_id","review_content","rating"])

# COMMAND ----------

# MAGIC %md
# MAGIC ##Rename review_content to review_body###

# COMMAND ----------

df = df.withColumnRenamed("review_content", "review_body") 


# COMMAND ----------

# MAGIC %md
# MAGIC ###Selecting only Review_Content & Sentiment columns###

# COMMAND ----------

df_final = df.select("review_body","sentiment")



# COMMAND ----------

# MAGIC %md
# MAGIC ###Removing Duplicates from the final review_body###

# COMMAND ----------

df_final = df_final.dropDuplicates(["review_body"])

# COMMAND ----------

# MAGIC %md
# MAGIC ###writing the Final Output into the Gold layer###

# COMMAND ----------

df_final.coalesce(1).write.mode("overwrite").option("header","true").csv("/Volumes/poc_amazon/gold/output/training_dataset/")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Rename part file to training_dataset.csv###

# COMMAND ----------

output_path = "/Volumes/poc_amazon/gold/output/training_dataset_temp/"

df_final.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(output_path)


# COMMAND ----------

import shutil
import os

# Paths
src_folder = output_path.replace("dbfs:", "/dbfs")
target_folder = "/Volumes/poc_amazon/gold/output/training_dataset/"
target_filename = "training_dataset.csv"

# Create target directory if not exists
os.makedirs(target_folder, exist_ok=True)

# Find the part file inside the temp directory
for file in os.listdir(src_folder):
    if file.startswith("part-") and file.endswith(".csv"):
        part_file_path = os.path.join(src_folder, file)
        target_file_path = os.path.join(target_folder, target_filename)

        # Move & rename
        shutil.move(part_file_path, target_file_path)
        break


# COMMAND ----------

# MAGIC %md
# MAGIC ##Step:5 Visualization of the Sentiment by Bar Diagram ##

# COMMAND ----------

# MAGIC %md
# MAGIC #Visualization the Sentiment_count#

# COMMAND ----------

from pyspark.sql.functions import count

sentiment_count = (
    df_final
    .groupBy("sentiment")
    .agg(count("*").alias("reviews"))
)


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

pdf = sentiment_count.toPandas()

pdf.plot(kind="bar", x="sentiment", y="reviews", figsize=(6,4))
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ##Step:6 Copying the Source file and training_dataset to Repo##

# COMMAND ----------

dbutils.fs.cp(
    "dbfs:/Volumes/poc_amazon/bronze/src/amazon.csv",
    "dbfs:/Workspace/Users/sam79.udutha@gmail.com/amazon_data_cleaning/raw_data/amazon.csv"
)


# COMMAND ----------

dbutils.fs.cp(
    "/Volumes/poc_amazon/gold/output/training_dataset/training_dataset.csv", "/Workspace/Users/sam79.udutha@gmail.com/amazon_data_cleaning/output")

# COMMAND ----------

dbutils.fs.cp("/Volumes/poc_amazon/gold/output/visualisation.jpg", "/Workspace/Users/sam79.udutha@gmail.com/amazon_data_cleaning/output")