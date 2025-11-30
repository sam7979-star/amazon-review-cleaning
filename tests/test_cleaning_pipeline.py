import pytest
from utils.transformations import clean_review
#Clean Review Function
def test_clean_review_basic():
    text = "Hello!!! Visit http://example.com NOW!!!"
    cleaned = clean_review(text)
    assert "http" not in cleaned
    assert "visit" in cleaned
    assert cleaned.islower()

def test_clean_review_handles_none():
    assert clean_review(None) is None

def test_clean_review_removes_html():
    text = "<p>This is <b>good</b></p>"
    cleaned = clean_review(text)
    assert "<" not in cleaned
    assert "good" in cleaned

#Test-2
#ID Filtering
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pytest

spark = SparkSession.builder.master("local[*]").appName("test").getOrCreate()

problem_ids = [
    "B088Z1YWBC", "B0BLV1GNLN", "B0BF57RN3K"
]

def test_problem_id_filtering():
    data = [("B088Z1YWBC","test"),("XYZ123","ok")]
    df = spark.createDataFrame(data, ["product_id","product_name"])

    df_problem = df.filter(col("product_id").isin(problem_ids))
    df_clean = df.filter(~col("product_id").isin(problem_ids))

    assert df_problem.count() == 1
    assert df_clean.count() == 1

#Test-3
#Sentiment Classification
from pyspark.sql.functions import col

def test_sentiment_assignment():
    data = [(1.5,),(3.5,),(4.5,),(None,)]
    df = spark.createDataFrame(data, ["rating"])

    df = df.withColumn(
        "sentiment",
        when(col("rating").isNull(), "unknown")
        .when((col("rating") >= 1.0) & (col("rating") < 3.0), "negative")
        .when((col("rating") >= 3.0) & (col("rating") < 4.0), "neutral")
        .when(col("rating") >= 4.0, "positive")
    )

    rows = [r.sentiment for r in df.collect()]
    assert rows == ["negative", "neutral", "positive", "unknown"]

