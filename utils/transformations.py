from pyspark.sql.functions import *
from pyspark.sql.types import *
import re
import unicodedata
# ======================================================
# UDF: Deep text cleaning
# ======================================================
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import re
import unicodedata

def clean_review(text):
    if text is None:
        return None

    text = str(text).strip()
    

    # Normalize unicode (removes accents, weird chars)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()

    # Remove HTML
    text = re.sub(r'<.*?>', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # Remove long repeated hyphens or punctuation (---, ----, :::::)
    text = re.sub(r'[-_=]{2,}', ' ', text)

    # Remove repeated punctuation sequences
    text = re.sub(r'[\.\,\;\:\?\/\\]{2,}', ' ', text)

    # Replace multiple spaces/dots/dashes between words
    text = re.sub(r'\s*[-]+\s*', ' ', text)

    # Remove digits (optional)
    text = re.sub(r'\b\d+\b', ' ', text)

    # Fix words stuck together (helloWorld → hello. world)
    text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)

    # Remove any remaining non-alphabetic noise
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove single characters (a, x, k, g)
    text = re.sub(r'\b[a-zA-Z]\b', ' ', text)

    # Collapse excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    ######
     # Remove long repeated characters (xxxxxx, -------, ::::::)
    text = re.sub(r"([x\-_:;=~])\1{2,}", " ", text)

    # Remove standalone non-words and repeated garbage segments
    #text = re.sub(r"\b[a-z]{1,2}\b", " ", text)  # single/double letters
    #text = re.sub(r"[^a-z\s]", " ", text)       # remove symbols/digits

    # Fix missing spaces between words
    text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)

    '''# Insert basic sentence breaks using heuristic cues
    text = re.sub(r"\bbut\b", ". but", text)
    text = re.sub(r"\band\b", " and ", text)
    text = re.sub(r"\bso\b", ". so", text)
    text = re.sub(r"\bthen\b", ". then", text)
    text = re.sub(r"\bhence\b", ". hence", text)
    text = re.sub(r"\bhowever\b", ". however", text)
    text = re.sub(r"\boverall\b", ". overall", text)
    text = re.sub(r"\bfinally\b", ". finally", text)'''

    # Reduce multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Final sentence segmentation enhancement
    text = re.sub(r"\s*\.\s*", ". ", text)

    return text.lower().strip()

clean_udf = udf(clean_review, StringType())
