import pandas as pd
import numpy as np

# Load data
print("📥 Loading data...")
interactions_df = pd.read_csv('sampled_interactions_df.csv')
reviews_df = pd.read_csv('GoodreadsData1/reviews.csv')

# Step 1: Filter reviews to valid structure
print("📝 Preparing reviews data...")
review_text_df = reviews_df[[
    'review_id', 'user_id', 'book_id', 'review_text',
    'date_added', 'date_updated', 'read_at', 'started_at',
    'n_votes', 'n_comments'
]].copy()

# Step 2: Prepare review ratings (without text metadata)
ratings_from_reviews = reviews_df.drop(columns=[
    'review_text', 'review_id', 'n_votes', 'n_comments'
]).copy()
ratings_from_reviews['is_reviewed'] = 1

# Step 3: Add is_reviewed = 0 to interaction data
interactions_df['is_reviewed'] = 0

# Step 4: Merge both into one ratings dataframe
ratings_df = pd.concat([ratings_from_reviews, interactions_df], ignore_index=True)

# Step 5: Impute missing dates using non-null review values
print("📅 Imputing missing dates...")

# Drop missing date rows from source and get numpy arrays
valid_date_added = reviews_df['date_added'].dropna().values
valid_read_at = reviews_df['read_at'].dropna().values

# Fill missing date_added
ratings_df['date_added'] = ratings_df['date_added'].fillna(
    pd.Series(np.random.choice(valid_date_added, size=len(ratings_df)), index=ratings_df.index)
)

# Fill missing read_at
ratings_df['read_at'] = ratings_df['read_at'].fillna(
    pd.Series(np.random.choice(valid_read_at, size=len(ratings_df)), index=ratings_df.index)
)

# Final cleanup
ratings_df = ratings_df.reset_index(drop=True)
review_text_df = review_text_df.reset_index(drop=True)

# ✅ Output datasets
ratings_df.to_csv('ratings_df.csv', index=False)
review_text_df.to_csv('reviews_df.csv', index=False)

print(f"✅ Exported ratings_df.csv with {len(ratings_df):,} rows")
print(f"✅ Exported reviews_df.csv with {len(review_text_df):,} reviews")
