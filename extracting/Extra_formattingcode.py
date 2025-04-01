import pandas as pd
import numpy as np
import os

# Paths
DATA_DIR = './GoodreadsData1'
INTERACTIONS_PATH = os.path.join(DATA_DIR, 'goodreads_interactions.csv')
BOOKS_PATH = os.path.join(DATA_DIR, 'books.csv')

# Load required columns only
print("📥 Loading data...")
interactions_df = pd.read_csv(
    INTERACTIONS_PATH,
    usecols=['user_id', 'book_id', 'is_read', 'rating', 'is_reviewed'],
    dtype={'user_id': int, 'book_id': int, 'is_read': int, 'rating': float, 'is_reviewed': int},
    low_memory=False
)

books_df = pd.read_csv(
    BOOKS_PATH,
    usecols=['book_id'],
    dtype={'book_id': int}
)

# Step 1: Filter to valid books AND rating != 0 AND is_reviewed == 0
print("⚙️ Filtering to rating ≠ 0 and is_reviewed == 0...")
valid_books = set(books_df['book_id'].unique())
filtered_df = interactions_df[
    (interactions_df['is_reviewed'] == 0) &
    (interactions_df['rating'] != 0) &
    (interactions_df['book_id'].isin(valid_books))
]

# Step 2: Get top users by number of valid interactions
print("🔍 Selecting top users...")
user_counts = filtered_df['user_id'].value_counts()
cumulative_sum = user_counts.cumsum()
top_user_ids = cumulative_sum[cumulative_sum <= 3_000_000].index

# Step 3: Filter to interactions from top users
sampled_interactions_df = filtered_df[filtered_df['user_id'].isin(top_user_ids)]

# Step 4: Trim down to exactly 3M if needed
if len(sampled_interactions_df) > 3_000_000:
    sampled_interactions_df = sampled_interactions_df.sample(n=3_000_000, random_state=42)

# Step 5: Export
output_path = 'sampled_interactions_df.csv'
sampled_interactions_df.to_csv(output_path, index=False)
print(f"✅ Saved {len(sampled_interactions_df):,} valid interactions to {output_path}")
