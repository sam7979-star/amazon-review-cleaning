**Amazon Review Cleaning & Sentiment Pipeline**


**📝 Project: The Messy Review Dataset Data Engineering**

This project converts a raw, messy Amazon product review CSV into a clean, curated, sentiment-labeled dataset suitable for training Machine Learning models.

The solution is implemented using PySpark on Databricks, following a Bronze → Silver → Gold medallion architecture.

                          ┌─────────────────────────┐
                          │    Desktop Source CSV    │
                          │      amazon.csv          │
                          └─────────────┬───────────┘
                                        │ Upload
                                        ▼
                     ┌──────────────────────────────────────┐
                     │              BRONZE LAYER             │
                     │ Raw ingestion (No cleaning applied)   │
                     │ /Volumes/poc_amazon/bronze/src/       │
                     └───────────────┬───────────────────────┘
                                     │
                                     │ Cleaning, parsing,
                                     │ fixing corrupt rows,
                                     │ exploding multi-review rows
                                     ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │                           SILVER LAYER                           │
        │  • Clean Unicode (₹ issues)                                      │
        │  • Normalize multiline reviews                                   │
        │  • Lowercasing, HTML removal, URL removal, emoji cleaning        │
        │  • Deduplicate using (user_id, product_id, review_content)       │
        │  • Convert rating → double                                       │
        │  • Create sentiment column                                       │
        │  /Volumes/poc_amazon/silver/clean_data/                          │
        └───────────────────────┬──────────────────────────────────────────┘
                                │
                                │ Select required columns
                                ▼
                    ┌──────────────────────────────────────────────┐
                    │                   GOLD LAYER                  │
                    │    Final Clean ML-ready dataset               │
                    │    Columns: review_body, sentiment            │
                    │    training_dataset.csv                       │
                    │ /Volumes/poc_amazon/gold/output/              │
                    └─────────────────────┬────────────────────────┘
                                          │ Visualization Step
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │   Sentiment Distribution Visualization        │
                   │   (matplotlib / pandas)                       │
                   │   • positive                                  │
                   │   • neutral                                   │
                   │   • negative                                  │
                   └──────────────────────────────────────────────┘

**_✅ Step 1: Data Ingestion into Bronze Layer_**

📌 What was done?

Uploaded the amazon.csv file from local desktop into Databricks using the UI and saved it into:

/Volumes/poc_amazon/bronze/src/

**📌 Why Bronze?**

Bronze layer stores raw, unmodified data exactly as received.

✔ Actions Performed

Loaded the file using spark.read.csv()

Enabled:

multiLine=True

quote="\""

escape="\""

mode="PERMISSIVE"

This ensured Spark could read long product descriptions, even if they contained commas.

Outcome:

Now have a Bronze raw dataset identical to the source (including corrupt rows).

**_✅ Step 2  Data Cleaning & Corruption Handling (Silver Layer)_**


**_🚨 Problems Found:_**

Corrupted product_name rows
Missing quotes caused entire columns to shift.

Review fields merged into arrays
Example: one row contained 8 reviews inside one line.

Unicode corruption (â‚¹ instead of ₹)

Empty or missing reviews

Duplicate reviews

Noise inside text:

HTML

URLs

emojis

repeated punctuation

non-alphabetic symbols

**_🛠 How  Fixed Each Issue_**


✔ 2.1 Re-Parsing Corrupt CSV Rows

Used:

.option("multiLine", True)
.option("quote", '"')
.option("escape", '"')


**_This fixed column-shift issues._**

**_✔ 2.2 Filtering known corrupted product_id_**

Manually identified 59 product_ids:

df_clean = df.filter(~col("product_id").isin(problem_ids))

**_✔ 2.3 Unicode Normalization_**

In transformations.clean_review():

text = unicodedata.normalize("NFKD", text)
text = text.encode("ascii", "ignore").decode()

**_✔ 2.4 Cleaning review text_**


lowercasing

removing HTML:

re.sub(r'<.*?>', ' ', text)


removing URLs

removing emoji

fixing whitespace

removing non-alphabetic characters

**_✔ 2.5 Removing useless reviews_**


df = df.filter(
    col("review_content").isNotNull() &
    (length(trim(col("review_content"))) > 0)
)


**_✔ 2.6 Deduplication (Silver Layer Pre-processing)_**


df = df.dropDuplicates(["user_id", "product_id", "review_content", "rating"])


**_📌 Outcome of Step 2_**


A completely normalized, cleaned, well-structured Silver dataset.


**_✅ Step 3 — Sentiment Label Creation_**

Sentiment is derived from the rating column.


**_⭐ Mapping Rules_**

| Rating    | Sentiment |
| --------- | --------- |
| 1.0 – 2.9 | negative  |
| 3.0 – 3.9 | neutral   |
| 4.0 – 5.0 | positive  |
| null      | unknown   |


**_✔ PySpark Code_**


df = df.withColumn(
    "sentiment",
    when(col("rating").isNull(), "unknown")
    .when((col("rating") >= 1.0) & (col("rating") < 3.0), "negative")
    .when((col("rating") >= 3.0) & (col("rating") < 4.0), "neutral")
    .when(col("rating") >= 4.0, "positive")
)


**_✅ Step 4: Sentiment Visualization_**

Generated a bar chart showing the count of each sentiment class.

**✔ Code**


sentiment_count = df_final.groupBy("sentiment").count()
pdf = sentiment_count.toPandas()

pdf.plot(kind="bar", x="sentiment", y="count", figsize=(6,4))
plt.title("Sentiment Distribution")
plt.show()


**_📊 Graph Interpretation_**


Positive reviews dominate (most Amazon electronics reviews are positive)

Negative reviews form the smallest group

Neutral ratings appear mid-range

This helps data scientists understand dataset balance for ML training.


**_🥇 Final Output (Gold Layer)_**


/Volumes/poc_amazon/gold/output/training_dataset/


**_📦 training_dataset.csv includes:_**


Column	Description
review_body	Cleaned textual review
sentiment	negative / neutral / positive


**_✔ Final Deduplication_**


df_final = df_final.dropDuplicates(["review_body"])


Ensures the dataset contains unique review text, which prevents ML model bias.

