import pandas as pd
import gzip
import json
import os

# File name (assumes it's in the same directory as this script)
INPUT_FILE = 'goodreads_book_authors.json.gz'
OUTPUT_FILE = 'authors.csv'

# Load all records from the compressed JSON file
with gzip.open(INPUT_FILE, 'rt', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# Convert to DataFrame
authors_df = pd.DataFrame(data)

# Optional: drop duplicates if any
authors_df.drop_duplicates(subset='author_id', inplace=True)

# Save to CSV
authors_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Extracted {len(authors_df)} unique authors to {OUTPUT_FILE}")
