"""Step 1: slim the 2 GB Indeed CSV down to the columns the analysis uses (about 1 minute).

Reads  ../indeed_job_listings.csv
Writes outputs/indeed_slim.parquet
"""
from pathlib import Path
import time
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "indeed_job_listings.csv"
DEST = HERE / "outputs" / "indeed_slim.parquet"
DEST.parent.mkdir(exist_ok=True)
COLS = ["jobid", "company_name", "date_posted_parsed", "job_title", "job_type", "location", "salary_formatted",
        "company_rating", "company_reviews_count", "country", "region", "description_text"]

t = time.time()
parts = []
for chunk in pd.read_csv(SRC, usecols=COLS, chunksize=100_000, low_memory=False):
    chunk["description_text"] = chunk["description_text"].astype(str).str[:3000]  # keep files small
    parts.append(chunk)
df = pd.concat(parts)
df.to_parquet(DEST)
print(f"{len(df):,} postings, {df.shape[1]} columns -> {DEST.name} in {time.time() - t:.0f}s")
