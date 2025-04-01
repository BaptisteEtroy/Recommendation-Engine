import os
import pandas as pd
import numpy as np
import re
import warnings
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import defaultdict
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Import sentiment analysis module
import sentiment_analysis

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

warnings.filterwarnings('ignore')

# Set data paths
DATA_DIR = './GoodreadsData1'

# Create a directory for saved data if it doesn't exist
SAVED_DATA_DIR = './saved_data'
os.makedirs(SAVED_DATA_DIR, exist_ok=True)

# Create visualization directories
VIZ_DIR = os.path.join(SAVED_DATA_DIR, 'visualizations')
os.makedirs(VIZ_DIR, exist_ok=True)

# Model-specific visualization directories
MODEL_VIZ_DIRS = {
    'general': os.path.join(VIZ_DIR, 'general'),
    'random': os.path.join(VIZ_DIR, 'random'),
    'popularity': os.path.join(VIZ_DIR, 'popularity'),
    'memory_cf': os.path.join(VIZ_DIR, 'memory_cf'),
    'model_cf': os.path.join(VIZ_DIR, 'model_cf'),
    'content_based': os.path.join(VIZ_DIR, 'content_based'),
    'context_aware': os.path.join(VIZ_DIR, 'context_aware'),
    'advanced_hybrid': os.path.join(VIZ_DIR, 'advanced_hybrid'),
    'comparison': os.path.join(VIZ_DIR, 'comparison')
}

# Create directories
for dir_path in MODEL_VIZ_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

def save_dataframe(df, filename):
    """Save DataFrame to pickle file."""
    filepath = os.path.join(SAVED_DATA_DIR, filename)
    df.to_pickle(filepath)
    print(f"Saved {filename} to {filepath}")
    return filepath

def load_dataframe(filename):
    """Load DataFrame from pickle file if it exists."""
    filepath = os.path.join(SAVED_DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Loading {filename} from {filepath}")
        return pd.read_pickle(filepath)
    return None

# Check if processed data already exists
processed_data_exists = all([
    os.path.exists(os.path.join(SAVED_DATA_DIR, 'reviews_df.pkl')),
    os.path.exists(os.path.join(SAVED_DATA_DIR, 'books_df.pkl')),
    os.path.exists(os.path.join(SAVED_DATA_DIR, 'ratings_df.pkl')),
    os.path.exists(os.path.join(SAVED_DATA_DIR, 'genres_df.pkl')),
    os.path.exists(os.path.join(SAVED_DATA_DIR, 'authors_df.pkl'))
])

# STEP 1: Load or read raw data
if processed_data_exists:
    print("Loading processed data from saved files...")
    reviews_df = load_dataframe('reviews_df.pkl')
    books_df = load_dataframe('books_df.pkl')
    ratings_df = load_dataframe('ratings_df.pkl')
    genres_df = load_dataframe('genres_df.pkl')
    authors_df = load_dataframe('authors_df.pkl')
else:
    print("Loading data from CSV files...")
    # Load essential datasets
    reviews_df = pd.read_csv(os.path.join(DATA_DIR, 'reviews.csv'))
    books_df = pd.read_csv(os.path.join(DATA_DIR, 'books.csv'))
    genres_df = pd.read_csv(os.path.join(DATA_DIR, 'genres.csv'))
    authors_df = pd.read_csv(os.path.join(DATA_DIR, 'authors.csv'))
    reviews_df = pd.read_csv(os.path.join(DATA_DIR, 'reviews_df.csv'))
    ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'ratings_df.csv'))
    
    # Save the raw data
    save_dataframe(reviews_df, 'reviews_df.pkl')
    save_dataframe(books_df, 'books_df.pkl')
    save_dataframe(ratings_df, 'ratings_df.pkl')
    save_dataframe(genres_df, 'genres_df.pkl')
    save_dataframe(authors_df, 'authors_df.pkl')

print(f"Reviews shape: {reviews_df.shape}")
print(f"Books shape: {books_df.shape}")
print(f"Genres shape: {genres_df.shape}")
print(f"Ratings shape: {ratings_df.shape}")

# Check if all sentiment data files exist
sentiment_data_exists = all([
    os.path.exists(os.path.join(SAVED_DATA_DIR, 'sentiment_scores.pkl')),
    os.path.exists(os.path.join(SAVED_DATA_DIR, 'book_sentiments.pkl')),
    os.path.exists(os.path.join(SAVED_DATA_DIR, 'user_sentiments.pkl'))
])

# Load or generate sentiment scores
if sentiment_data_exists:
    print("Loading sentiment data from saved files...")
    sentiment_df = pd.read_pickle(os.path.join(SAVED_DATA_DIR, 'sentiment_scores.pkl'))
    book_sentiments = pd.read_pickle(os.path.join(SAVED_DATA_DIR, 'book_sentiments.pkl'))
    user_sentiments = pd.read_pickle(os.path.join(SAVED_DATA_DIR, 'user_sentiments.pkl'))
    print(f"Loaded sentiment data for {len(sentiment_df)} reviews")
else:
    print("Generating sentiment scores from review text...")
    try:
        # Run the sentiment analysis module on the reviews
        sentiment_df, book_sentiments, user_sentiments = sentiment_analysis.run_sentiment_analysis(reviews_df)
        print(f"Generated sentiment scores for {len(sentiment_df)} reviews")
        
        # Explicitly save all data files
        if not os.path.exists(os.path.join(SAVED_DATA_DIR, 'sentiment_scores.pkl')):
            sentiment_df.to_pickle(os.path.join(SAVED_DATA_DIR, 'sentiment_scores.pkl'))
        if not os.path.exists(os.path.join(SAVED_DATA_DIR, 'book_sentiments.pkl')):
            book_sentiments.to_pickle(os.path.join(SAVED_DATA_DIR, 'book_sentiments.pkl'))
        if not os.path.exists(os.path.join(SAVED_DATA_DIR, 'user_sentiments.pkl')):
            user_sentiments.to_pickle(os.path.join(SAVED_DATA_DIR, 'user_sentiments.pkl'))
    except Exception as e:
        print(f"Warning: Sentiment analysis failed with error: {e}")
        print("Continuing without sentiment analysis...")
        # Create empty dataframes to avoid further errors
        sentiment_df = pd.DataFrame(columns=['user_id', 'book_id', 'sentiment_score', 'sentiment_normalized'])
        book_sentiments = pd.DataFrame(columns=['book_id', 'sentiment_mean', 'sentiment_std', 'review_count', 'sentiment_norm_mean'])
        user_sentiments = pd.DataFrame(columns=['user_id', 'sentiment_mean', 'sentiment_std', 'review_count', 'sentiment_norm_mean'])

# Check if preprocessed data exists
preprocessed_data_exists = all([
    os.path.exists(os.path.join(SAVED_DATA_DIR, 'ratings_df_preprocessed.pkl')),
    os.path.exists(os.path.join(SAVED_DATA_DIR, 'reviews_df_preprocessed.pkl'))
])

# STEP 2: Basic preprocessing
if preprocessed_data_exists:
    print("Loading preprocessed data...")
    ratings_df = load_dataframe('ratings_df_preprocessed.pkl')
    reviews_df = load_dataframe('reviews_df_preprocessed.pkl')
else:
    print("Performing basic preprocessing...")
    # Filter zero ratings
    ratings_df = ratings_df[ratings_df["rating"] != 0]
    # Fix negative comment counts
    reviews_df['n_comments'] = reviews_df['n_comments'].abs()
    
    # Save preprocessed data
    save_dataframe(ratings_df, 'ratings_df_preprocessed.pkl')
    save_dataframe(reviews_df, 'reviews_df_preprocessed.pkl')

# Genre processing functions
def safe_parse(genre_str):
    """Safely parse genre string to dictionary."""
    if pd.isna(genre_str) or genre_str == '':
        return {}
    try:
        # Clean the string of non-standard JSON characters
        genre_str = re.sub(r"'", '"', genre_str)
        genre_str = genre_str.replace("None", '"None"')
        genre_str = "{" + genre_str + "}"
        return literal_eval(genre_str)
    except:
        return {}

def extract_genres(genre_dict):
    """Extract genres from genre dictionary."""
    if not genre_dict:
        return []
    genres = []
    for k, v in genre_dict.items():
        if k and k != 'None':
            genres.append(k.lower().strip())
    return genres

def extract_shelves(shelf_str):
    """Extract shelf names from shelf string."""
    if pd.isna(shelf_str) or shelf_str == '':
        return []
    
    shelves = []
    pairs = shelf_str.split(';')
    for pair in pairs:
        if ':' in pair:
            # Split on first colon only, in case there are multiple colons
            parts = pair.split(':', 1)
            name = parts[0]
            shelves.append(name.lower().strip())
    return shelves

def shelf_vector(shelf_list):
    """Create a binary vector from shelf list."""
    shelf_dict = {
        'to-read': 0, 'read': 0, 'currently-reading': 0,
        'favorites': 0, 'owned': 0
    }
    
    for shelf in shelf_list:
        for key in shelf_dict.keys():
            if key in shelf:
                shelf_dict[key] = 1
                break
    
    return list(shelf_dict.values())

# Check if feature-engineered data exists
feature_data_exists = os.path.exists(os.path.join(SAVED_DATA_DIR, 'books_df_with_features.pkl'))

# STEP 3: Genre and shelf processing
if feature_data_exists:
    print("Loading feature-engineered data...")
    books_df = load_dataframe('books_df_with_features.pkl')
else:
    print("Processing genre data...")
    books_df['genre_dict'] = books_df['genre'].apply(safe_parse)
    books_df['genre_list'] = books_df['genre_dict'].apply(extract_genres)

    print("Processing shelf data...")
    books_df['shelf_list'] = books_df['popular_shelves'].apply(extract_shelves)
    books_df['shelf_vector'] = books_df['shelf_list'].apply(shelf_vector)
    
    # Save feature-engineered books data
    save_dataframe(books_df, 'books_df_with_features.pkl')

# Check if context data exists
context_data_exists = os.path.exists(os.path.join(SAVED_DATA_DIR, 'ratings_df_with_context.pkl'))

# STEP 4: Add time context features
if context_data_exists:
    print("Loading ratings with context features...")
    ratings_df = load_dataframe('ratings_df_with_context.pkl')
else:
    print("Adding time context features...")
    # Check if date_added column exists
    if 'date_added' in ratings_df.columns:
        # Try to convert date strings to datetime objects with more explicit error handling
        try:
            ratings_df['date_added'] = pd.to_datetime(ratings_df['date_added'], errors='coerce')
            
            # Only create time features if we have valid datetime data
            if not ratings_df['date_added'].isna().all():
                # Extract temporal features
                ratings_df['month'] = ratings_df['date_added'].dt.month
                ratings_df['day_of_week'] = ratings_df['date_added'].dt.dayofweek
                ratings_df['hour'] = ratings_df['date_added'].dt.hour
                
                # Create time period categorical feature
                def get_time_period(hour):
                    if pd.isna(hour):
                        return np.nan
                    if 6 <= hour < 12:
                        return 'morning'
                    elif 12 <= hour < 18:
                        return 'afternoon'
                    elif 18 <= hour < 22:
                        return 'evening'
                    else:
                        return 'night'
                
                ratings_df['time_period'] = ratings_df['hour'].apply(get_time_period)
            else:
                print("Warning: No valid datetime data in 'date_added' column")
                # Add placeholder columns to avoid errors later
                ratings_df['month'] = 0
                ratings_df['day_of_week'] = 0
                ratings_df['hour'] = 0
                ratings_df['time_period'] = 'unknown'
        except Exception as e:
            print(f"Warning: Could not convert 'date_added' to datetime: {e}")
            # Add placeholder columns to avoid errors later
            ratings_df['month'] = 0
            ratings_df['day_of_week'] = 0
            ratings_df['hour'] = 0
            ratings_df['time_period'] = 'unknown'
    else:
        print("Warning: 'date_added' column not found in ratings_df")
        # Add placeholder columns to avoid errors later
        ratings_df['month'] = 0
        ratings_df['day_of_week'] = 0
        ratings_df['hour'] = 0
        ratings_df['time_period'] = 'unknown'
    
    # Save ratings with context features
    save_dataframe(ratings_df, 'ratings_df_with_context.pkl')

# STEP 5: Model data preparation
sample_data_exists = all([
    os.path.exists(os.path.join(SAVED_DATA_DIR, 'sample_ratings.pkl')),
    os.path.exists(os.path.join(SAVED_DATA_DIR, 'train_df.pkl')),
    os.path.exists(os.path.join(SAVED_DATA_DIR, 'test_df.pkl'))
])

def get_sample_and_split():
    # Get sample for demonstration
    sample_ratings = ratings_df.sample(frac=0.004, random_state=42)  # sample size of .4%, modify as needed
    save_dataframe(sample_ratings, 'sample_ratings.pkl')
    print(f"Working with {len(sample_ratings)} ratings sample")
    
    # Split the data
    train_df, test_df = train_test_split_by_user(sample_ratings, test_size=0.2)
    save_dataframe(train_df, 'train_df.pkl')
    save_dataframe(test_df, 'test_df.pkl')
    print(f"Train set: {len(train_df)} ratings, Test set: {len(test_df)} ratings")
    
    return sample_ratings, train_df, test_df

# -------------- METRICS FUNCTIONS ----------------

def train_test_split_by_user(ratings, test_size=0.2, random_state=42):
    """Split ratings data by user for training and testing."""
    np.random.seed(random_state)
    
    # Initialize train and test sets
    train_ratings = pd.DataFrame()
    test_ratings = pd.DataFrame()
    
    # Group by user
    user_groups = ratings.groupby('user_id')
    
    for user_id, group in user_groups:
        # For each user, split their ratings
        n_test = max(1, int(len(group) * test_size))
        
        # Randomly select test items
        test_indices = np.random.choice(group.index, size=n_test, replace=False)
        test_group = ratings.loc[test_indices]
        train_group = ratings.loc[~ratings.index.isin(test_indices) & (ratings['user_id'] == user_id)]
        
        # Add to train and test sets
        train_ratings = pd.concat([train_ratings, train_group])
        test_ratings = pd.concat([test_ratings, test_group])
    
    return train_ratings, test_ratings

def get_explicit_metrics(y_true, y_pred, verbose=False):
    """Calculate explicit feedback metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    if verbose:
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    
    return {"RMSE": rmse, "MAE": mae}

def get_implicit_metrics(y_true, y_pred):
    """Calculate implicit feedback metrics."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = 0
    
    return {"Precision": precision, "Recall": recall, "F1": f1, "AUC": auc}

def mean_reciprocal_rank(relevant_items, predictions_ranking):
    """Calculate Mean Reciprocal Rank."""
    ranks = []
    for user_rel, user_pred in zip(relevant_items, predictions_ranking):
        # Find positions of relevant items in pred list
        for rel_item in user_rel:
            if rel_item in user_pred:
                # +1 because ranks start at 1
                ranks.append(1.0 / (user_pred.index(rel_item) + 1))
                break
            else:
                ranks.append(0)
    
    return np.mean(ranks) if ranks else 0

def get_ranking_metrics(relevant_items, pred_items, k=10):
    """
    Calculate ranking metrics like Precision@k, Recall@k, and MAP@k.
    
    Args:
        relevant_items: List of lists, where each inner list contains relevant items for a user
        pred_items: List of lists, where each inner list contains predicted items for a user
        k: The cutoff for evaluation
    
    Returns:
        Dictionary of ranking metrics
    """
    # Precision@k
    precision_at_k = []
    for rel, pred in zip(relevant_items, pred_items):
        # Count number of relevant items in top-k predictions
        if len(pred[:k]) > 0:
            precision_at_k.append(len(set(rel) & set(pred[:k])) / len(pred[:k]))
        else:
            precision_at_k.append(0)
    
    # Recall@k
    recall_at_k = []
    for rel, pred in zip(relevant_items, pred_items):
        if len(rel) > 0:
            recall_at_k.append(len(set(rel) & set(pred[:k])) / len(rel))
        else:
            recall_at_k.append(0)
    
    # MAP@k (Mean Average Precision)
    map_at_k = []
    for rel, pred in zip(relevant_items, pred_items):
        if not rel:  # Skip if no relevant items
            continue
            
        avg_precision = 0
        hits = 0
        
        for i, item in enumerate(pred[:k]):
            if item in rel:
                hits += 1
                precision_at_i = hits / (i + 1)
                avg_precision += precision_at_i
        
        if len(rel) > 0:
            map_at_k.append(avg_precision / min(len(rel), k))
        else:
            map_at_k.append(0)
    
    # NDCG@k (Normalized Discounted Cumulative Gain)
    ndcg_at_k = []
    for rel, pred in zip(relevant_items, pred_items):
        if not rel:  # Skip if no relevant items
            continue
            
        # Calculate DCG@k
        dcg = 0
        for i, item in enumerate(pred[:k]):
            if item in rel:
                # Using binary relevance (1 if relevant, 0 if not)
                # Position is i+1 because i starts at 0
                dcg += 1 / np.log2(i + 2)  # log base 2 of position + 1
        
        # Calculate ideal DCG@k (IDCG@k)
        # Sort items by relevance (all relevant items come first)
        idcg = 0
        for i in range(min(len(rel), k)):
            idcg += 1 / np.log2(i + 2)
        
        # Calculate NDCG@k
        if idcg > 0:
            ndcg_at_k.append(dcg / idcg)
        else:
            ndcg_at_k.append(0)
    
    return {
        f"Precision@{k}": np.mean(precision_at_k) if precision_at_k else 0,
        f"Recall@{k}": np.mean(recall_at_k) if recall_at_k else 0,
        f"MAP@{k}": np.mean(map_at_k) if map_at_k else 0,
        f"NDCG@{k}": np.mean(ndcg_at_k) if ndcg_at_k else 0
    }

def to_implicit(r, threshold=4):
    """Convert explicit ratings to implicit."""
    return 1 if r >= threshold else 0

def get_top_n(predictions, n=10, solve_ties=False):
    """Get top N predictions for each user."""
    top_n = {}
    for user_id, item_id, _, est, _ in predictions:
        if user_id not in top_n:
            top_n[user_id] = []
        top_n[user_id].append((item_id, est))
    
    for user_id, ratings in top_n.items():
        if solve_ties:
            # Add small random values to break ties
            ratings = [(i, r + np.random.normal(0, 1e-6)) for i, r in ratings]
        
        ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[user_id] = ratings[:n]
    
    return top_n

# -------------- MODELS ----------------

def run_random_recommender(train_df, test_df):
    print("\n---------- Running Random Recommender (Baseline) ----------")
    
    # Global mean rating from train set
    global_mean = train_df['rating'].mean()
    
    # Generate random predictions for test set
    np.random.seed(42)
    predictions_random = np.random.uniform(1, 5, size=len(test_df))
    
    # Evaluate the results
    explicit_results = get_explicit_metrics(test_df['rating'].values, predictions_random, verbose=True)
    
    # Implicit evaluation
    # Convert to binary ratings (positive if >= 4, negative otherwise)
    y_true = test_df['is_reviewed'].values  # Assume this is already binary
    y_pred = np.array([1 if r >= 4 else 0 for r in predictions_random])
    
    implicit_results = get_implicit_metrics(y_true, y_pred)
    print("Random Model - Implicit Evaluation Results:")
    for metric, value in implicit_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize rating distribution
    plot_rating_distribution(predictions_random, "Random Model - Rating Distribution", 'random')
    
    # Visualize confusion matrix
    plot_confusion_matrix(y_true, y_pred, "Random Model - Confusion Matrix", 'random')
    
    # Coverage evaluation
    all_book_ids = set(train_df['book_id'].unique()) | set(test_df['book_id'].unique())
    recommended_book_ids = set(test_df['book_id'])  # Random can recommend any book
    
    coverage_results = get_coverage_metrics(all_book_ids, recommended_book_ids)
    print("Random Model - Coverage Evaluation Results:")
    for metric, value in coverage_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Combine results
    results = {**explicit_results, **implicit_results, **coverage_results}
    return results

def run_popularity_recommender(train_df, test_df):
    print("\n---------- Running Popularity-Based Recommender (Baseline) ----------")
    
    # Calculate global mean
    global_mean = train_df['rating'].mean()
    
    # Calculate mean rating for each book from the training set
    book_mean_ratings = train_df.groupby('book_id')['rating'].mean()
    
    # Count ratings for each book
    rating_counts = train_df['book_id'].value_counts()
    
    # Set minimum ratings threshold as median
    m = rating_counts.median()
    C = global_mean
    
    # Calculate weighted ratings (Bayesian average)
    weighted_ratings = {}
    for book_id in book_mean_ratings.index:
        R = book_mean_ratings[book_id]  # Average rating
        v = rating_counts.get(book_id, 0)  # Number of ratings
        # Apply formula: WR = (v / (v + m)) * R + (m / (v + m)) * C
        weighted_rating = (v / (v + m)) * R + (m / (v + m)) * C
        weighted_ratings[book_id] = weighted_rating
    
    # Make predictions for test set
    predictions_popularity = []
    for _, row in test_df.iterrows():
        book_id = row['book_id']
        # Use the book's weighted rating or global mean if no data
        pred_rating = weighted_ratings.get(book_id, global_mean)
        predictions_popularity.append(pred_rating)
    
    # Evaluate the results
    explicit_results = get_explicit_metrics(test_df['rating'].values, predictions_popularity, verbose=True)
    
    # Implicit evaluation
    y_true = test_df['is_reviewed'].values
    y_pred = np.array([1 if r >= 4 else 0 for r in predictions_popularity])
    
    implicit_results = get_implicit_metrics(y_true, y_pred)
    print("Popularity Model - Implicit Evaluation Results:")
    for metric, value in implicit_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize rating distribution
    plot_rating_distribution(predictions_popularity, "Popularity Model - Rating Distribution", 'popularity')
    
    # Visualize confusion matrix
    plot_confusion_matrix(y_true, y_pred, "Popularity Model - Confusion Matrix", 'popularity')
    
    # Visualize top books by weighted rating
    plt.figure(figsize=(12, 6))
    top_books = sorted(weighted_ratings.items(), key=lambda x: x[1], reverse=True)[:20]
    book_ids, ratings = zip(*top_books)
    
    # Get titles for top books
    book_titles = []
    for book_id in book_ids:
        title = books_df[books_df['book_id'] == book_id]['title'].values
        if len(title) > 0:
            # Truncate long titles
            title_str = str(title[0])
            book_titles.append(title_str[:20] + '...' if len(title_str) > 20 else title_str)
        else:
            book_titles.append(f"Book {book_id}")
    
    plt.barh(range(len(book_titles)), ratings, align='center')
    plt.yticks(range(len(book_titles)), book_titles)
    plt.xlabel('Weighted Rating')
    plt.title('Top 20 Popular Books by Weighted Rating')
    plt.gca().invert_yaxis()  # Display the highest rated at the top
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_VIZ_DIRS['popularity'], 'top_popular_books.png'))
    plt.close()
    
    # Coverage evaluation
    all_book_ids = set(train_df['book_id'].unique()) | set(test_df['book_id'].unique())
    recommended_book_ids = set(weighted_ratings.keys())
    
    coverage_results = get_coverage_metrics(all_book_ids, recommended_book_ids)
    print("Popularity Model - Coverage Evaluation Results:")
    for metric, value in coverage_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Combine results
    results = {**explicit_results, **implicit_results, **coverage_results}
    return results

def run_all_models():
    """Run all recommendation models and evaluate them."""
    # Create empty dictionary to store models results
    results = {}
    
    # Check if sample ratings exist
    sample_ratings_path = os.path.join(SAVED_DATA_DIR, 'sample_ratings.pkl')
    train_df_path = os.path.join(SAVED_DATA_DIR, 'train_df.pkl')
    test_df_path = os.path.join(SAVED_DATA_DIR, 'test_df.pkl')
    
    if all([os.path.exists(p) for p in [sample_ratings_path, train_df_path, test_df_path]]):
        print("Loading existing sample and train/test split...")
        sample_ratings = pd.read_pickle(sample_ratings_path)
        train_df = pd.read_pickle(train_df_path)
        test_df = pd.read_pickle(test_df_path)
    else:
        print("Creating new sample and train/test split...")
        sample_ratings, train_df, test_df = get_sample_and_split()
        
        # Save to disk for future runs
        sample_ratings.to_pickle(sample_ratings_path)
        train_df.to_pickle(train_df_path)
        test_df.to_pickle(test_df_path)
    
    print(f"Sample size: {len(sample_ratings)} ratings")
    print(f"Train set: {len(train_df)} ratings")
    print(f"Test set: {len(test_df)} ratings")
    
    # Display sample statistics
    print("\nSample Statistics:")
    print(f"Unique users: {sample_ratings['user_id'].nunique()}")
    print(f"Unique books: {sample_ratings['book_id'].nunique()}")
    print(f"Rating distribution:\n{sample_ratings['rating'].value_counts().sort_index()}")
    
    # Visualize the rating distribution in the sample
    plt.figure(figsize=(10, 6))
    sample_ratings['rating'].value_counts().sort_index().plot(kind='bar')
    plt.title('Rating Distribution in Sample')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(MODEL_VIZ_DIRS['general'], 'sample_rating_distribution.png'))
    plt.close()
    
    # Run Random Recommender (as baseline)
    results['Random'] = run_random_recommender(train_df, test_df)
    
    # Run Popularity-Based Recommender (as baseline)
    results['Popularity'] = run_popularity_recommender(train_df, test_df)
    
    # Run Memory-based Collaborative Filtering
    results['Memory-based CF'] = run_cf_memory_based(train_df, test_df)
    
    # Run Model-based Collaborative Filtering
    results['Model-based CF'] = run_cf_model_based(train_df, test_df)
    
    # Run Content-Based Recommender
    results['Content-based'] = run_content_based(train_df, test_df)
    
    # Run Context-Aware Recommender
    results['Context-aware'] = run_context_aware(train_df, test_df)
    
    # Run Advanced Hybrid Model
    results['Advanced Hybrid'] = run_advanced_hybrid_model(train_df, test_df)
    
    # Create summary visualizations comparing all models
    # RMSE Comparison
    visualize_model_comparison(results, 'RMSE')
    
    # MAE Comparison
    visualize_model_comparison(results, 'MAE')
    
    # Precision Comparison (if available)
    if 'Precision' in list(results.values())[0]:
        visualize_model_comparison(results, 'Precision')
    
    # Recall Comparison (if available)
    if 'Recall' in list(results.values())[0]:
        visualize_model_comparison(results, 'Recall')
    
    # F1 Comparison (if available)
    if 'F1' in list(results.values())[0]:
        visualize_model_comparison(results, 'F1')
    
    # Catalog_Coverage Comparison
    if 'Catalog_Coverage' in list(results.values())[0]:
        visualize_model_comparison(results, 'Catalog_Coverage')
    
    # Visualize radar chart of all metrics for comparison
    # Get common metrics across all models
    all_metrics = set.intersection(*[set(m.keys()) for m in results.values()])
    metrics_to_plot = [m for m in all_metrics if m not in ['RMSE', 'MAE']]
    
    if metrics_to_plot:
        # Create radar chart for classifier metrics
        n_metrics = len(metrics_to_plot)
        angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Use a list of colors instead of 'viridis'
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        for i, (model_name, model_results) in enumerate(results.items()):
            values = [model_results[metric] for metric in metrics_to_plot]
            values += values[:1]  # Close the polygon
            
            color_idx = i % len(colors)  # Cycle through colors if more models than colors
            ax.plot(angles, values, linewidth=2, label=model_name, color=colors[color_idx])
            ax.fill(angles, values, alpha=0.1, color=colors[color_idx])
        
        # Set labels and layout
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_to_plot)
        ax.set_ylim(0, 1)
        plt.title('Model Comparison on Implicit Metrics', size=15)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_VIZ_DIRS['comparison'], 'model_comparison_radar.png'))
        plt.close()
    
    # Print overall summary
    print("\n=== OVERALL MODEL COMPARISON ===")
    print("Explicit Feedback Metrics (lower is better):")
    print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10}")
    print("-" * 40)
    for model, metrics in results.items():
        print(f"{model:<20} {metrics['RMSE']:<10.4f} {metrics['MAE']:<10.4f}")
    
    # Print implicit metrics and include coverage
    if all('Precision' in metrics for metrics in results.values()):
        print("\nImplicit Feedback & Coverage Metrics (higher is better):")
        print(f"{'Model':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Coverage':<10}")
        print("-" * 60)
        for model, metrics in results.items():
            print(f"{model:<20} {metrics['Precision']:<10.4f} {metrics['Recall']:<10.4f} {metrics['F1']:<10.4f} {metrics['Catalog_Coverage']:<10.4f}")
    
    # Save results to a pickle file
    with open(os.path.join(SAVED_DATA_DIR, 'model_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    return results

def predict_rating_cf(user_id, item_id, user_item_matrix, item_similarity, book_id_to_idx, global_mean):
    """Collaborative filtering prediction function."""
    if user_id not in user_item_matrix.index or item_id not in book_id_to_idx:
        return global_mean
    
    user_ratings = user_item_matrix.loc[user_id].values
    item_idx = book_id_to_idx[item_id]
    
    # Get similarity scores
    sim_scores = item_similarity[item_idx]
    
    # Calculate weighted sum
    weighted_sum = 0
    sim_sum = 0
    
    for j, rating in enumerate(user_ratings):
        if not np.isnan(rating) and j != item_idx:
            weighted_sum += rating * sim_scores[j]
            sim_sum += abs(sim_scores[j])
    
    if sim_sum == 0:
        return global_mean
    
    return weighted_sum / sim_sum

def run_cf_memory_based(train_df, test_df):
    print("\n---------- Running Memory-based Collaborative Filtering ----------")
    
    # Get sample book IDs for memory efficiency
    sample_book_ids = set(train_df['book_id'].unique()) | set(test_df['book_id'].unique())
    train_df_filtered = train_df[train_df['book_id'].isin(sample_book_ids)]
    
    # Build User-Item Matrix from train_df for CF
    user_item_matrix = train_df_filtered.pivot_table(
        index='user_id',
        columns='book_id',
        values='rating',
        aggfunc='mean'
    )
    
    # Print shape of user-item matrix
    print(f"User-Item Matrix Shape: {user_item_matrix.shape}")
    
    # Calculate global mean
    global_mean = train_df_filtered['rating'].mean()
    
    # Recommended users need at least this many ratings to give good recs
    min_ratings = 2
    
    # Fill missing values with 0 for similarity computation
    user_item_filled = user_item_matrix.fillna(0)
    
    # Calculate item-item similarity
    print("Calculating item-item similarity matrix...")
    item_similarity = cosine_similarity(user_item_filled.T)
    print(f"Item-Item Similarity Matrix Shape: {item_similarity.shape}")
    
    # Mapping book IDs to matrix indices for fast lookup
    book_id_to_idx = {book_id: i for i, book_id in enumerate(user_item_matrix.columns)}
    
    # Make predictions for test set
    print("Making predictions for test set...")
    predictions_cf = []
    
    # Tracking variables for coverage
    recommended_book_ids = set()
    
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        book_id = row['book_id']
        
        # Predict rating
        pred_rating = predict_rating_cf(user_id, book_id, user_item_matrix, 
                                       item_similarity, book_id_to_idx, global_mean)
        predictions_cf.append(pred_rating)
        
        # Track book IDs that are being recommended (if prediction is >= 3.5)
        if pred_rating >= 3.5:
            recommended_book_ids.add(book_id)
    
    # Evaluate the results
    explicit_results = get_explicit_metrics(test_df['rating'].values, predictions_cf, verbose=True)
    
    # Implicit evaluation
    y_true = test_df['is_reviewed'].values
    y_pred = np.array([1 if r >= 4 else 0 for r in predictions_cf])
    
    implicit_results = get_implicit_metrics(y_true, y_pred)
    print("Memory-based CF - Implicit Evaluation Results:")
    for metric, value in implicit_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize rating distribution
    plot_rating_distribution(predictions_cf, "Memory-based CF - Rating Distribution", 'memory_cf')
    
    # Visualize confusion matrix
    plot_confusion_matrix(y_true, y_pred, "Memory-based CF - Confusion Matrix", 'memory_cf')
    
    # Visualize similarity heatmap for a sample of items
    if len(item_similarity) > 10:
        # Take a small sample for better visualization
        sample_size = min(20, len(item_similarity))
        sample_indices = np.random.choice(len(item_similarity), sample_size, replace=False)
        sample_similarity = item_similarity[sample_indices][:, sample_indices]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(sample_similarity, cmap='viridis', annot=False)
        plt.title('Item-Item Similarity Heatmap (Sample)')
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_VIZ_DIRS['memory_cf'], 'item_similarity_heatmap.png'))
        plt.close()
    
    # Coverage evaluation
    all_book_ids = set(train_df['book_id'].unique()) | set(test_df['book_id'].unique())
    
    coverage_results = get_coverage_metrics(all_book_ids, recommended_book_ids)
    print("Memory-based CF - Coverage Evaluation Results:")
    for metric, value in coverage_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Combine results
    results = {**explicit_results, **implicit_results, **coverage_results}
    return results

def run_cf_model_based(train_df, test_df):
    print("\n---------- Running Model-based Collaborative Filtering ----------")
    
    # Convert to Surprise format
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df[['user_id', 'book_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    # Train SVD model
    print("Training SVD model...")
    algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    algo.fit(trainset)
    
    # Make predictions for test set
    print("Making predictions for test set...")
    predictions_svd = []
    
    # For tracking coverage
    recommended_book_ids = set()
    
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        book_id = row['book_id']
        
        try:
            pred = algo.predict(user_id, book_id)
            predictions_svd.append(pred.est)
            
            # Track book IDs that are being recommended (if prediction is >= 3.5)
            if pred.est >= 3.5:
                recommended_book_ids.add(book_id)
                
        except Exception as e:
            # If prediction fails, use the global mean
            global_mean = train_df['rating'].mean()
            predictions_svd.append(global_mean)
    
    # Evaluate the results
    explicit_results = get_explicit_metrics(test_df['rating'].values, predictions_svd, verbose=True)
    
    # Implicit evaluation
    y_true = test_df['is_reviewed'].values
    y_pred = np.array([1 if r >= 4 else 0 for r in predictions_svd])
    
    implicit_results = get_implicit_metrics(y_true, y_pred)
    print("Model-based CF - Implicit Evaluation Results:")
    for metric, value in implicit_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize rating distribution
    plot_rating_distribution(predictions_svd, "Model-based CF - Rating Distribution", 'model_cf')
    
    # Visualize confusion matrix
    plot_confusion_matrix(y_true, y_pred, "Model-based CF - Confusion Matrix", 'model_cf')
    
    # Visualize true vs predicted ratings
    plt.figure(figsize=(10, 8))
    plt.scatter(test_df['rating'].values, predictions_svd, alpha=0.5)
    plt.plot([1, 5], [1, 5], 'r--')  # Perfect prediction line
    plt.xlabel('True Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Model-based CF: True vs Predicted Ratings')
    plt.xlim(0.5, 5.5)
    plt.ylim(0.5, 5.5)
    plt.savefig(os.path.join(MODEL_VIZ_DIRS['model_cf'], 'true_vs_pred.png'))
    plt.close()
    
    # Coverage evaluation
    all_book_ids = set(train_df['book_id'].unique()) | set(test_df['book_id'].unique())
    
    coverage_results = get_coverage_metrics(all_book_ids, recommended_book_ids)
    print("Model-based CF - Coverage Evaluation Results:")
    for metric, value in coverage_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Combine results
    results = {**explicit_results, **implicit_results, **coverage_results}
    return results

def run_content_based(train_df, test_df):
    print("\n---------- Running Content-Based Recommender ----------")
    
    # Get sample book IDs
    sample_book_ids = set(train_df['book_id'].unique()) | set(test_df['book_id'].unique())
    
    # Filter books to include only those in sample
    sample_books = books_df[books_df['book_id'].isin(sample_book_ids)].copy()
    
    # Feature preparation: combine title, genres and description
    sample_books['content'] = sample_books['title'].fillna('') + ' ' + \
                             sample_books['genre_list'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '') + ' ' + \
                             sample_books['description'].fillna('')
    
    # Create TF-IDF vectors
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(sample_books['content'])
    
    # Calculate cosine similarity
    content_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Mapping book IDs to matrix indices
    book_id_to_idx = {book_id: i for i, book_id in enumerate(sample_books['book_id'])}
    
    # Global mean for fallback
    global_mean = train_df['rating'].mean()
    
    # Define prediction function
    def predict_rating_content(user_id, candidate_item_id):
        """
        Predict rating for a user-item pair using content-based filtering
        """
        # Get user's ratings
        user_ratings = train_df[train_df['user_id'] == user_id]
        
        if len(user_ratings) == 0 or candidate_item_id not in book_id_to_idx:
            return global_mean
        
        item_idx = book_id_to_idx[candidate_item_id]
        
        # Calculate weighted sum of ratings
        weighted_sum = 0
        sim_sum = 0
        
        for _, row in user_ratings.iterrows():
            rated_item_id = row['book_id']
            
            if rated_item_id in book_id_to_idx:
                rated_idx = book_id_to_idx[rated_item_id]
                sim = content_similarity[item_idx, rated_idx]
                
                weighted_sum += row['rating'] * sim
                sim_sum += abs(sim)
        
        if sim_sum == 0:
            return global_mean
        
        return weighted_sum / sim_sum
    
    # Make predictions for test set
    print("Making content-based predictions for test set...")
    predictions_content = []
    
    # For tracking coverage
    recommended_book_ids = set()
    
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        book_id = row['book_id']
        
        # Predict rating
        pred_rating = predict_rating_content(user_id, book_id)
        predictions_content.append(pred_rating)
        
        # Track book IDs that are being recommended (if prediction is >= 3.5)
        if pred_rating >= 3.5:
            recommended_book_ids.add(book_id)
    
    # Evaluate the results
    explicit_results = get_explicit_metrics(test_df['rating'].values, predictions_content, verbose=True)
    
    # Implicit evaluation
    y_true = test_df['is_reviewed'].values
    y_pred = np.array([1 if r >= 4 else 0 for r in predictions_content])
    
    implicit_results = get_implicit_metrics(y_true, y_pred)
    print("Content-based - Implicit Evaluation Results:")
    for metric, value in implicit_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize rating distribution
    plot_rating_distribution(predictions_content, "Content-based - Rating Distribution", 'content_based')
    
    # Visualize confusion matrix
    plot_confusion_matrix(y_true, y_pred, "Content-based - Confusion Matrix", 'content_based')
    
    # Visualize content similarity heatmap for a sample of items
    if len(content_similarity) > 10:
        # Take a small sample for better visualization
        sample_size = min(20, len(content_similarity))
        sample_indices = np.random.choice(len(content_similarity), sample_size, replace=False)
        sample_similarity = content_similarity[sample_indices][:, sample_indices]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(sample_similarity, cmap='viridis', annot=False)
        plt.title('Content Similarity Heatmap (Sample)')
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_VIZ_DIRS['content_based'], 'content_similarity_heatmap.png'))
        plt.close()
    
    # Visualize true vs predicted ratings
    plt.figure(figsize=(10, 8))
    plt.scatter(test_df['rating'].values, predictions_content, alpha=0.5)
    plt.plot([1, 5], [1, 5], 'r--')  # Perfect prediction line
    plt.xlabel('True Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Content-based: True vs Predicted Ratings')
    plt.xlim(0.5, 5.5)
    plt.ylim(0.5, 5.5)
    plt.savefig(os.path.join(MODEL_VIZ_DIRS['content_based'], 'true_vs_pred.png'))
    plt.close()
    
    # Coverage evaluation
    all_book_ids = set(train_df['book_id'].unique()) | set(test_df['book_id'].unique())
    
    coverage_results = get_coverage_metrics(all_book_ids, recommended_book_ids)
    print("Content-based - Coverage Evaluation Results:")
    for metric, value in coverage_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Combine results
    results = {**explicit_results, **implicit_results, **coverage_results}
    return results

def run_context_aware(train_df, test_df):
    print("\n---------- Running Context-Aware Recommender ----------")
    
    # Get sample book IDs
    sample_book_ids = set(train_df['book_id'].unique()) | set(test_df['book_id'].unique())
    
    # Filter books to include only those in sample
    sample_books = books_df[books_df['book_id'].isin(sample_book_ids)].copy()
    
    # Feature preparation: combine title, genres and description
    sample_books['content'] = sample_books['title'].fillna('') + ' ' + \
                             sample_books['genre_list'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '') + ' ' + \
                             sample_books['description'].fillna('')
    
    # Create TF-IDF vectors
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(sample_books['content'])
    
    # Calculate cosine similarity
    content_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Mapping book IDs to matrix indices
    book_id_to_idx = {book_id: i for i, book_id in enumerate(sample_books['book_id'])}
    
    # Global mean for fallback
    global_mean = train_df['rating'].mean()
    
    # Compute context bias
    context_bias = {}
    context_values = train_df['day_of_week'].unique()
    
    for context_value in context_values:
        context_mean = train_df[train_df['day_of_week'] == context_value]['rating'].mean()
        context_bias[context_value] = context_mean - global_mean
    
    # Define content-based prediction function
    def predict_rating_content(user_id, candidate_item_id):
        """
        Predict rating for a user-item pair using content-based filtering
        """
        # Get user's ratings
        user_ratings = train_df[train_df['user_id'] == user_id]
        
        if len(user_ratings) == 0 or candidate_item_id not in book_id_to_idx:
            return global_mean
        
        item_idx = book_id_to_idx[candidate_item_id]
        
        # Calculate weighted sum of ratings
        weighted_sum = 0
        sim_sum = 0
        
        for _, row in user_ratings.iterrows():
            rated_item_id = row['book_id']
            
            if rated_item_id in book_id_to_idx:
                rated_idx = book_id_to_idx[rated_item_id]
                sim = content_similarity[item_idx, rated_idx]
                
                weighted_sum += row['rating'] * sim
                sim_sum += abs(sim)
        
        if sim_sum == 0:
            return global_mean
        
        return weighted_sum / sim_sum
    
    # Define context-aware prediction function
    def predict_rating_context_aware(user_id, candidate_item_id, context_value):
        """
        Predict rating using content-based filtering with context bias
        """
        # Get base content prediction
        base_pred = predict_rating_content(user_id, candidate_item_id)
        
        # Apply context bias
        if context_value in context_bias:
            return base_pred + context_bias[context_value]
        
        return base_pred
    
    # Make predictions for test set
    print("Making context-aware predictions for test set...")
    predictions_context = []
    
    # For tracking coverage
    recommended_book_ids = set()
    
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        book_id = row['book_id']
        context_value = row['day_of_week']  # Using day of week as context
        
        # Predict rating
        pred_rating = predict_rating_context_aware(user_id, book_id, context_value)
        predictions_context.append(pred_rating)
        
        # Track book IDs that are being recommended (if prediction is >= 3.5)
        if pred_rating >= 3.5:
            recommended_book_ids.add(book_id)
    
    # Evaluate the results
    explicit_results = get_explicit_metrics(test_df['rating'].values, predictions_context, verbose=True)
    
    # Implicit evaluation
    y_true = test_df['is_reviewed'].values
    y_pred = np.array([1 if r >= 4 else 0 for r in predictions_context])
    
    implicit_results = get_implicit_metrics(y_true, y_pred)
    print("Context-aware - Implicit Evaluation Results:")
    for metric, value in implicit_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize rating distribution
    plot_rating_distribution(predictions_context, "Context-aware - Rating Distribution", 'context_aware')
    
    # Visualize confusion matrix
    plot_confusion_matrix(y_true, y_pred, "Context-aware - Confusion Matrix", 'context_aware')
    
    # Visualize true vs predicted ratings
    plt.figure(figsize=(10, 8))
    plt.scatter(test_df['rating'].values, predictions_context, alpha=0.5)
    plt.plot([1, 5], [1, 5], 'r--')  # Perfect prediction line
    plt.xlabel('True Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Context-aware: True vs Predicted Ratings')
    plt.xlim(0.5, 5.5)
    plt.ylim(0.5, 5.5)
    plt.savefig(os.path.join(MODEL_VIZ_DIRS['context_aware'], 'true_vs_pred.png'))
    plt.close()
    
    # Visualize context bias
    plt.figure(figsize=(10, 6))
    context_labels = list(context_bias.keys())
    bias_values = list(context_bias.values())
    
    bars = plt.bar(context_labels, bias_values, color='skyblue')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('Context (Day of Week)')
    plt.ylabel('Rating Bias')
    plt.title('Context Bias by Day of Week')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(MODEL_VIZ_DIRS['context_aware'], 'context_bias.png'))
    plt.close()
    
    # Coverage evaluation
    all_book_ids = set(train_df['book_id'].unique()) | set(test_df['book_id'].unique())
    
    coverage_results = get_coverage_metrics(all_book_ids, recommended_book_ids)
    print("Context-aware - Coverage Evaluation Results:")
    for metric, value in coverage_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Combine results
    results = {**explicit_results, **implicit_results, **coverage_results}
    return results

def run_advanced_hybrid_model(train_df, test_df):
    print("\n---------- Running Advanced Hybrid Recommender (Multiple Models) ----------")
    
    # Initialize component tracking dictionaries
    component_counts = {
        'memory_cf': 0,
        'model_cf': 0,
        'content': 0,
        'context': 0, 
        'popularity': 0,
        'sentiment': 0
    }
    
    component_weights = {
        'memory_cf': 0,
        'model_cf': 0,
        'content': 0,
        'context': 0,
        'popularity': 0,
        'sentiment': 0
    }
    
    # Load sentiment profiles for books and users if available
    book_sentiments_path = os.path.join(SAVED_DATA_DIR, 'book_sentiments.pkl')
    user_sentiments_path = os.path.join(SAVED_DATA_DIR, 'user_sentiments.pkl')
    
    if os.path.exists(book_sentiments_path) and os.path.exists(user_sentiments_path):
        book_sentiments = pd.read_pickle(book_sentiments_path)
        user_sentiments = pd.read_pickle(user_sentiments_path)
        
        # Create lookup dictionaries for faster access
        book_sentiment_dict = dict(zip(book_sentiments['book_id'], book_sentiments['sentiment_norm_mean']))
        user_sentiment_dict = dict(zip(user_sentiments['user_id'], user_sentiments['sentiment_norm_mean']))
        
        print(f"Loaded sentiment profiles for {len(book_sentiment_dict)} books and {len(user_sentiment_dict)} users")
        
        # Include sentiment in feature engineering
        use_sentiment = True
    else:
        use_sentiment = False
        print("Sentiment data not available")
    
    # --- First, set up Memory-based CF ---
    print("Setting up Memory-based CF component...")
    sample_book_ids = set(train_df['book_id'].unique()) | set(test_df['book_id'].unique())
    train_df_filtered = train_df[train_df['book_id'].isin(sample_book_ids)]
    
    # Build User-Item Matrix from train_df for CF
    user_item_matrix = train_df_filtered.pivot_table(
        index='user_id',
        columns='book_id',
        values='rating',
        aggfunc='mean'
    )
    
    user_item_filled = user_item_matrix.fillna(0)
    item_similarity_cf = cosine_similarity(user_item_filled.T)
    
    global_mean = train_df_filtered['rating'].mean()
    book_id_to_idx_cf = {book_id: i for i, book_id in enumerate(user_item_matrix.columns)}
    
    # --- Set up Model-based CF (SVD) ---
    print("Setting up Model-based CF (SVD) component...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df[['user_id', 'book_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    svd_model.fit(trainset)
    
    # --- Set up Content-Based ---
    print("Setting up Content-based component...")
    sample_books = books_df[books_df['book_id'].isin(sample_book_ids)].copy()
    
    # Feature preparation: combine title, genres and description
    sample_books['content'] = sample_books['title'].fillna('') + ' ' + \
                             sample_books['genre_list'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '') + ' ' + \
                             sample_books['description'].fillna('')
    
    # Create TF-IDF vectors
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(sample_books['content'])
    
    # Calculate cosine similarity between books
    content_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Mapping book IDs to matrix indices
    book_id_to_idx_content = {book_id: i for i, book_id in enumerate(sample_books['book_id'])}
    
    # --- Set up Context-Aware ---
    print("Setting up Context-aware component...")
    # Compute context bias
    context_bias = {}
    context_values = train_df['day_of_week'].unique()
    
    for context_value in context_values:
        context_mean = train_df[train_df['day_of_week'] == context_value]['rating'].mean()
        context_bias[context_value] = context_mean - global_mean
    
    # --- Set up Popularity-Based ---
    print("Setting up Popularity-based component...")
    # Calculate weighted ratings (bayesian average)
    book_avg_ratings = train_df.groupby('book_id')['rating'].mean()
    rating_counts = train_df['book_id'].value_counts()
    
    # Set minimum threshold as median number of ratings
    m = rating_counts.median()
    
    # Calculate weighted ratings
    weighted_ratings = {}
    for book_id in sample_book_ids:
        if book_id in book_avg_ratings.index:
            R = book_avg_ratings[book_id]  # Average rating
            v = rating_counts.get(book_id, 0)  # Number of ratings
            # Apply formula: WR = (v / (v + m)) * R + (m / (v + m)) * C
            weighted_rating = (v / (v + m)) * R + (m / (v + m)) * global_mean
            weighted_ratings[book_id] = weighted_rating
        else:
            weighted_ratings[book_id] = global_mean  # Use global mean for books without ratings
    
    # --- Define prediction functions for each component ---
    def predict_memory_cf(user_id, item_id):
        """Memory-based CF prediction function."""
        if user_id not in user_item_matrix.index or item_id not in book_id_to_idx_cf:
            return global_mean
        
        user_ratings = user_item_matrix.loc[user_id].values
        item_idx = book_id_to_idx_cf[item_id]
        
        # Get similarity scores
        sim_scores = item_similarity_cf[item_idx]
        
        # Calculate weighted sum
        weighted_sum = 0
        sim_sum = 0
        
        for j, rating in enumerate(user_ratings):
            if not np.isnan(rating) and j != item_idx:
                weighted_sum += rating * sim_scores[j]
                sim_sum += abs(sim_scores[j])
        
        if sim_sum == 0:
            return global_mean
        
        return weighted_sum / sim_sum
    
    def predict_model_cf(user_id, item_id):
        """Model-based CF prediction function using SVD."""
        try:
            pred = svd_model.predict(user_id, item_id).est
            return pred
        except:
            return global_mean
    
    def predict_content_based(user_id, item_id):
        """Content-based prediction function."""
        # Get user's ratings
        user_ratings = train_df[train_df['user_id'] == user_id]
        
        if len(user_ratings) == 0 or item_id not in book_id_to_idx_content:
            return global_mean
        
        item_idx = book_id_to_idx_content[item_id]
        
        weighted_sum = 0
        sim_sum = 0
        
        for _, row in user_ratings.iterrows():
            rated_item_id = row['book_id']
            if rated_item_id in book_id_to_idx_content:
                rated_idx = book_id_to_idx_content[rated_item_id]
                sim = content_similarity[item_idx, rated_idx]
                
                weighted_sum += row['rating'] * sim
                sim_sum += abs(sim)
        
        if sim_sum == 0:
            return global_mean
        
        return weighted_sum / sim_sum
    
    def predict_context_aware(user_id, item_id, context_value):
        """Context-aware prediction function."""
        # Get base content prediction
        base_pred = predict_content_based(user_id, item_id)
        
        # Apply context bias
        if context_value in context_bias:
            return base_pred + context_bias[context_value]
        return base_pred
    
    def predict_popularity(item_id):
        """Popularity-based prediction function."""
        return weighted_ratings.get(item_id, global_mean)
    
    # --- Define dynamic weighting function ---
    def calculate_weights(user_id, book_id, context_value):
        """
        Calculate dynamic weights for each model based on user and item characteristics.
        Returns a dictionary of weights that sum to 1.0
        """
        weights = {
            'memory_cf': 0.0,
            'model_cf': 0.0,
            'content': 0.0, 
            'context': 0.0,
            'popularity': 0.0
        }
        
        # Default weights (if no special cases apply)
        weights['memory_cf'] = 0.25
        weights['model_cf'] = 0.30
        weights['content'] = 0.20
        weights['context'] = 0.15
        weights['popularity'] = 0.10
        
        # --- Check for cold-start users (few ratings) ---
        user_ratings = train_df[train_df['user_id'] == user_id]
        rating_count = len(user_ratings)
        
        if rating_count < 5:  # Cold-start user
            # Reduce weight of CF methods, increase content and popularity
            weights['memory_cf'] *= 0.3
            weights['model_cf'] *= 0.5
            weights['content'] *= 1.5
            weights['context'] *= 1.3
            weights['popularity'] *= 2.0
        elif rating_count > 50:  # Power user with many ratings
            # Increase CF methods, decrease popularity
            weights['memory_cf'] *= 1.3
            weights['model_cf'] *= 1.2
            weights['popularity'] *= 0.5
        
        # --- Check for cold-start items (few ratings) ---
        item_ratings = train_df[train_df['book_id'] == book_id]
        item_rating_count = len(item_ratings)
        
        if item_rating_count < 3:  # Cold-start item
            # Reduce CF weights, increase content
            weights['memory_cf'] *= 0.5
            weights['model_cf'] *= 0.6
            weights['content'] *= 1.8
            weights['popularity'] *= 0.8
        
        # --- Check for strong context effect ---
        if context_value in context_bias and abs(context_bias[context_value]) > 0.2:
            # If context has strong effect, increase its weight
            weights['context'] *= 1.5
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            for k in weights:
                weights[k] /= weight_sum
        
        return weights
    
    # --- Advanced hybrid prediction function ---
    def predict_advanced_hybrid(user_id, book_id, context_value):
        """
        Advanced hybrid prediction function that intelligently combines
        various recommendation methods with dynamic weighting.
        """
        # Use the component tracking dictionaries from the outer function
        nonlocal component_counts, component_weights
        
        # Track which components are used
        components = []  # Actual predictions or scores
        weights = []     # Weights for each component
        explanations = []  # Explanation for each component
        
        # --- 1. Try Memory-based CF ---
        cf_memory_pred = None
        if user_id in user_avg and book_id in item_avg:
            neighbors = get_user_neighbors(user_id, 10)
            if neighbors:
                cf_memory_pred = memory_cf_predict(user_id, book_id, neighbors)
                # Confidence weight based on number of neighbors
                cf_confidence = min(1.0, len(neighbors) / 5)
                
                components.append(cf_memory_pred)
                weights.append(0.35 * cf_confidence)
                explanations.append(f"Memory CF: {cf_memory_pred:.2f} (weight: {0.35 * cf_confidence:.2f})")
                
                # Update component counts
                component_counts['memory_cf'] += 1
                component_weights['memory_cf'] += 0.35 * cf_confidence
        
        # --- 2. Try Model-based CF ---
        try:
            cf_model_pred = svd_model.predict(user_id, book_id).est
            components.append(cf_model_pred)
            weights.append(0.30)
            explanations.append(f"Model CF: {cf_model_pred:.2f} (weight: 0.30)")
            
            # Update component counts
            component_counts['model_cf'] += 1
            component_weights['model_cf'] += 0.30
        except:
            cf_model_pred = None
        
        # --- 3. Try Content-based ---
        if book_id in book_tfidf_indices:
            content_pred = content_predict(user_id, book_id)
            
            # Adjust weight based on how many books the user has rated
            user_book_count = len(user_ratings.get(user_id, []))
            content_confidence = min(1.0, user_book_count / 10)
            
            components.append(content_pred)
            weights.append(0.20 * content_confidence)
            explanations.append(f"Content: {content_pred:.2f} (weight: {0.20 * content_confidence:.2f})")
            
            # Update component counts
            component_counts['content'] += 1
            component_weights['content'] += 0.20 * content_confidence
        
        # --- 4. Try Popularity Baseline ---
        if book_id in item_avg:
            pop_pred = item_avg[book_id]
            components.append(pop_pred)
            weights.append(0.10)  # Low weight for popularity
            explanations.append(f"Popularity: {pop_pred:.2f} (weight: 0.10)")
            
            # Update component counts
            component_counts['popularity'] += 1
            component_weights['popularity'] += 0.10
        
        # --- 5. Try Context-awareness ---
        if context_value is not None:
            day_offset = 0.2  # Maximum influence of day of week
            
            # Convert context value to int (0-6 for days of week)
            day_idx = int(context_value)
            
            # Get day-specific adjustment from global patterns
            if day_idx in day_adjustments:
                # Scale to a small adjustment value
                context_adj = day_adjustments[day_idx] * day_offset
                
                components.append(3.5 + context_adj)  # 3.5 is the midpoint
                weights.append(0.05)  # Very low weight
                explanations.append(f"Context: {3.5 + context_adj:.2f} (weight: 0.05)")
                
                # Update component counts
                component_counts['context'] += 1
                component_weights['context'] += 0.05
        
        # --- 6. Add sentiment similarity component if available ---
        if 'book_sentiments' in globals() and 'user_sentiments' in globals():
            # Create dictionaries for faster lookups if not already created
            if 'book_sentiment_dict' not in globals():
                global book_sentiment_dict, user_sentiment_dict
                book_sentiment_dict = dict(zip(book_sentiments['book_id'], book_sentiments['sentiment_norm_mean']))
                user_sentiment_dict = dict(zip(user_sentiments['user_id'], user_sentiments['sentiment_norm_mean']))
            
            # Get book sentiment
            book_sentiment = book_sentiment_dict.get(book_id, 0.5)
            
            # Get user sentiment profile
            user_sentiment = user_sentiment_dict.get(user_id, 0.5)
            
            # Calculate sentiment compatibility (how close user's average sentiment is to book's sentiment)
            sentiment_compat = 1 - abs(book_sentiment - user_sentiment)
            
            # Convert to rating scale (1-5)
            sentiment_score = 1 + sentiment_compat * 4
            
            # Add sentiment component
            if book_sentiment != 0.5 and user_sentiment != 0.5:  # Only use if we have real data
                components.append(sentiment_score)
                weights.append(0.20)  # Give sentiment a moderate weight
                explanations.append(f"Sentiment: {sentiment_score:.2f} (weight: 0.20)")
                
                # Capture component in total
                component_counts['sentiment'] += 1
                component_weights['sentiment'] += 0.20
        
        # Default prediction if no components are available
        if not components:
            return 3.5  # Return middle value
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        norm_weights = [w/total_weight for w in weights]
        
        # Generate weighted prediction
        pred_rating = sum(c * w for c, w in zip(components, norm_weights))
        
        # Ensure prediction is in valid range
        pred_rating = max(1, min(5, pred_rating))
        
        return pred_rating
    
    # --- Make predictions for test set ---
    print("Making advanced hybrid predictions for test set...")
    predictions_hybrid = []
    user_ids = []
    book_ids = []
    true_ratings = []
    contexts = []
    
    # Track individual model predictions for analysis
    component_preds = {
        'memory_cf': [],
        'model_cf': [],
        'content': [],
        'context': [],
        'popularity': []
    }
    
    component_weight_lists = {
        'memory_cf': [],
        'model_cf': [],
        'content': [],
        'context': [],
        'popularity': [],
        'sentiment': []
    }
    
    # For tracking coverage
    recommended_book_ids = set()
    
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        book_id = row['book_id']
        context_value = row['day_of_week']  # Use day of week as context
        true_rating = row['rating']
        
        # Get individual component predictions
        memory_cf_pred = predict_memory_cf(user_id, book_id)
        model_cf_pred = predict_model_cf(user_id, book_id)
        content_pred = predict_content_based(user_id, book_id)
        context_pred = predict_context_aware(user_id, book_id, context_value)
        popularity_pred = predict_popularity(book_id)
        
        # Get weights for this prediction
        weights = calculate_weights(user_id, book_id, context_value)
        
        # Calculate weighted prediction
        pred_rating = (
            weights['memory_cf'] * memory_cf_pred +
            weights['model_cf'] * model_cf_pred +
            weights['content'] * content_pred +
            weights['context'] * context_pred +
            weights['popularity'] * popularity_pred
        )
        
        # Add to results
        predictions_hybrid.append(pred_rating)
        user_ids.append(user_id)
        book_ids.append(book_id)
        true_ratings.append(true_rating)
        contexts.append(context_value)
        
        # Store component predictions
        component_preds['memory_cf'].append(memory_cf_pred)
        component_preds['model_cf'].append(model_cf_pred)
        component_preds['content'].append(content_pred)
        component_preds['context'].append(context_pred)
        component_preds['popularity'].append(popularity_pred)
        
        # Store weights
        for k, v in weights.items():
            if k in component_weight_lists:
                component_weight_lists[k].append(v)
        
        # Add sentiment weight of 0 if not used in this prediction
        if 'sentiment' not in weights:
            component_weight_lists['sentiment'].append(0)
    
        # Track book IDs that are being recommended (if prediction is >= 3.5)
        if pred_rating >= 3.5:
            recommended_book_ids.add(book_id)
    
    # --- Evaluate the results ---
    explicit_results_hybrid = get_explicit_metrics(true_ratings, predictions_hybrid, verbose=True)
    
    # --- Visualization: Predicted vs Actual Ratings ---
    plt.figure(figsize=(10, 8))
    plt.scatter(true_ratings, predictions_hybrid, alpha=0.5)
    plt.plot([1, 5], [1, 5], 'r--')  # Perfect prediction line
    plt.xlabel('True Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Advanced Hybrid Model: True vs Predicted Ratings')
    plt.xlim(0.5, 5.5)
    plt.ylim(0.5, 5.5)
    plt.savefig(os.path.join(MODEL_VIZ_DIRS['advanced_hybrid'], 'true_vs_pred.png'))
    plt.close()
    
    # --- Visualization: Distribution of Predicted Ratings ---
    plot_rating_distribution(predictions_hybrid, "Distribution of Advanced Hybrid Model Predictions", 'advanced_hybrid')
    
    # --- Visualization: Component contribution ---
    avg_weights = {k: np.mean(v) for k, v in component_weight_lists.items()}
    
    plt.figure(figsize=(10, 6))
    plt.bar(list(avg_weights.keys()), list(avg_weights.values()), color='skyblue')
    plt.title('Average Component Weights in Advanced Hybrid Model')
    plt.ylabel('Weight')
    plt.ylim(0, 0.5)  # Set limit for better visualization
    
    # Add value labels on top of bars
    for i, (k, v) in enumerate(avg_weights.items()):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.savefig(os.path.join(MODEL_VIZ_DIRS['advanced_hybrid'], 'component_weights.png'))
    plt.close()
    
    # --- Compare individual component performance ---
    component_metrics = {}
    
    for component, preds in component_preds.items():
        metrics = get_explicit_metrics(true_ratings, preds, verbose=False)
        component_metrics[component] = metrics
    
    # Add hybrid results
    component_metrics['advanced_hybrid'] = explicit_results_hybrid
    
    # Create comparison bar chart for RMSE
    plt.figure(figsize=(12, 6))
    components = list(component_metrics.keys())
    rmse_values = [metrics['RMSE'] for metrics in component_metrics.values()]
    
    # Sort by RMSE (ascending)
    sorted_components = [x for _, x in sorted(zip(rmse_values, components))]
    sorted_rmse = sorted(rmse_values)
    
    bars = plt.bar(sorted_components, sorted_rmse, color='lightblue')
    plt.ylabel('RMSE (lower is better)')
    plt.title('RMSE Comparison: Advanced Hybrid vs. Component Models')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.savefig(os.path.join(MODEL_VIZ_DIRS['advanced_hybrid'], 'component_comparison.png'))
    plt.close()
    
    # --- Implicit Evaluation ---
    # Use existing implicit feedback columns
    y_true_hybrid = test_df['is_reviewed'].values
    y_pred_hybrid = np.array([1 if r >= 4 else 0 for r in predictions_hybrid])
    
    implicit_results_hybrid = get_implicit_metrics(y_true_hybrid, y_pred_hybrid)
    print("Advanced Hybrid Model - Implicit Evaluation Results:")
    for metric, value in implicit_results_hybrid.items():
        print(f"{metric}: {value:.4f}")
    
    # --- Visualization: Confusion Matrix for Implicit Task ---
    plot_confusion_matrix(y_true_hybrid, y_pred_hybrid, "Advanced Hybrid Model Confusion Matrix", 'advanced_hybrid')
    
    # --- Find top recommendations for sample users ---
    # Create dataframe with predictions
    pred_df = pd.DataFrame({
        'user_id': user_ids,
        'book_id': book_ids,
        'true_rating': true_ratings,
        'pred_rating': predictions_hybrid
    })
    
    # Sample a few users who have at least 5 predictions
    user_pred_counts = pred_df['user_id'].value_counts()
    sample_users = user_pred_counts[user_pred_counts >= 5].index[:3]
    
    for user_id in sample_users:
        # Get top 5 predictions for this user
        user_preds = pred_df[pred_df['user_id'] == user_id].sort_values('pred_rating', ascending=False)
        top_books = user_preds['book_id'].values[:5]
        
        # Display top book recommendations
        display_book_recommendations(user_id, top_books, books_df, 
                                     "Top 5 Advanced Hybrid Model Recommendations")
    
    # Coverage evaluation
    all_book_ids = set(train_df['book_id'].unique()) | set(test_df['book_id'].unique())
    
    coverage_results = get_coverage_metrics(all_book_ids, recommended_book_ids)
    print("Advanced Hybrid Model - Coverage Evaluation Results:")
    for metric, value in coverage_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Combine explicit and implicit results
    results = {**explicit_results_hybrid, **implicit_results_hybrid, **coverage_results}
    
    # Visualize sentiment distribution for books
    if 'book_sentiments' in globals():
        plt.figure(figsize=(10, 6))
        sns.histplot(book_sentiments['sentiment_mean'], kde=True)
        plt.axvline(0, color='r', linestyle='--')
        plt.title('Distribution of Book Sentiment Scores')
        plt.xlabel('Sentiment Score (-1 to 1)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(MODEL_VIZ_DIRS['advanced_hybrid'], 'book_sentiment_dist.png'))
        plt.close()
        
        # Plot relationship between sentiment and ratings
        plt.figure(figsize=(10, 6))
        sentiment_ratings = pd.merge(
            sentiment_df, 
            ratings_df[['user_id', 'book_id', 'rating']], 
            on=['user_id', 'book_id']
        )
        
        plt.scatter(sentiment_ratings['sentiment_score'], 
                   sentiment_ratings['rating'], 
                   alpha=0.3)
        plt.xlabel('Review Sentiment Score')
        plt.ylabel('Explicit Rating')
        plt.title('Relationship Between Review Sentiment and Rating')
        plt.savefig(os.path.join(MODEL_VIZ_DIRS['advanced_hybrid'], 'sentiment_vs_rating.png'))
        plt.close()
        
        # Plot most polarizing books (high sentiment variance)
        plt.figure(figsize=(12, 8))
        most_polarizing = book_sentiments.sort_values('sentiment_std', ascending=False).head(20)
        
        # Get book titles for these books
        book_titles = []
        for book_id in most_polarizing['book_id']:
            title = books_df[books_df['book_id'] == book_id]['title'].values
            if len(title) > 0:
                book_titles.append(title[0][:30] + '...' if len(title[0]) > 30 else title[0])
            else:
                book_titles.append(f"Book {book_id}")
        
        most_polarizing['short_title'] = book_titles
        
        # Plot with error bars
        plt.figure(figsize=(14, 10))
        plt.errorbar(
            x=range(len(most_polarizing)),
            y=most_polarizing['sentiment_mean'],
            yerr=most_polarizing['sentiment_std'],
            fmt='o',
            capsize=5
        )
        plt.xticks(range(len(most_polarizing)), most_polarizing['short_title'], rotation=90)
        plt.title("Most Polarizing Books by Sentiment")
        plt.xlabel("Book")
        plt.ylabel("Average Sentiment")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_VIZ_DIRS['advanced_hybrid'], 'polarizing_books.png'))
        plt.close()

    # Update component metrics plot to include sentiment
    if 'sentiment' in component_counts:
        component_counts_list = [component_counts.get(c, 0) for c in ['memory_cf', 'model_cf', 'content', 'context', 'popularity', 'sentiment']]
        normalized_weights = [component_weights.get(c, 0)/max(1, component_counts.get(c, 0)) for c in ['memory_cf', 'model_cf', 'content', 'context', 'popularity', 'sentiment']]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(['Memory CF', 'Model CF', 'Content', 'Context', 'Popularity', 'Sentiment'], 
                normalized_weights)
        plt.title('Average Component Weights in Advanced Hybrid Model')
        plt.ylabel('Average Weight')
        
        # Add count annotations
        for i, count in enumerate(component_counts_list):
            plt.text(i, normalized_weights[i] + 0.01, f'Used: {count}', ha='center')
            
        plt.savefig(os.path.join(MODEL_VIZ_DIRS['advanced_hybrid'], 'component_weights.png'))
        plt.close()

    # Create a report on how sentiment affects recommendations
    if 'sentiment_df' in globals():
        try:
            # Find correlation between sentiment and rating
            merged_data = pd.merge(
                sentiment_df[['user_id', 'book_id', 'sentiment_score']], 
                ratings_df[['user_id', 'book_id', 'rating']], 
                on=['user_id', 'book_id']
            )
            
            correlation = merged_data['sentiment_score'].corr(merged_data['rating'])
            
            # Add sentiment information to results
            results['sentiment_correlation'] = correlation
            results['sentiment_count'] = len(sentiment_df)
            
            print(f"\nSentiment Analysis Results:")
            print(f"- Correlation between review sentiment and rating: {correlation:.4f}")
            print(f"- Total reviews with sentiment analyzed: {len(sentiment_df)}")
            
            if 'component_counts' in globals() and 'sentiment' in component_counts:
                sentiment_usage = component_counts['sentiment'] / sum(test_df['is_reviewed'])
                print(f"- Sentiment component used in {sentiment_usage:.2%} of predictions")
        except Exception as e:
            print(f"Error generating sentiment report: {e}")
    
    return results

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', model_dir='general'):
    """Plot confusion matrix for implicit feedback evaluation."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(MODEL_VIZ_DIRS[model_dir], f"confusion_matrix.png"))
    plt.close()

def plot_rating_distribution(predictions, title='Rating Distribution', model_dir='general'):
    """Plot distribution of predicted ratings."""
    plt.figure(figsize=(10, 6))
    sns.histplot(predictions, bins=20, kde=True)
    plt.axvline(np.mean(predictions), color='red', linestyle='--', label=f'Mean: {np.mean(predictions):.2f}')
    plt.title(title)
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(MODEL_VIZ_DIRS[model_dir], f"rating_distribution.png"))
    plt.close()

def visualize_model_comparison(models_metrics, metric_name='RMSE'):
    """Visualize comparison of models based on a specific metric"""
    plt.figure(figsize=(12, 6))
    
    # Extract metric values for each model
    models = list(models_metrics.keys())
    metric_values = [metrics[metric_name] for metrics in models_metrics.values()]
    
    # Determine if lower is better (RMSE, MAE) or higher is better (all other metrics)
    lower_is_better = metric_name in ['RMSE', 'MAE']
    
    # Sort by metric value
    if lower_is_better:
        # Sort ascending for metrics where lower is better
        sorted_models = [x for _, x in sorted(zip(metric_values, models))]
        sorted_values = sorted(metric_values)
    else:
        # Sort descending for metrics where higher is better
        sorted_models = [x for _, x in sorted(zip(metric_values, models), reverse=True)]
        sorted_values = sorted(metric_values, reverse=True)
    
    # Create bar chart
    bars = plt.bar(sorted_models, sorted_values, color='cornflowerblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.title(f'Model Comparison by {metric_name}')
    plt.ylabel(metric_name)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if lower_is_better:
        plt.ylim(0, max(metric_values) * 1.1)
        plt.figtext(0.5, 0.01, 'Lower is better', ha='center')
    else:
        plt.ylim(0, min(metric_values) * 0.9)
        plt.figtext(0.5, 0.01, 'Higher is better', ha='center')
    
    plt.savefig(os.path.join(MODEL_VIZ_DIRS['comparison'], f"model_comparison_{metric_name.lower()}.png"))
    plt.close()

def get_top_n_recommendations(model_name, user_id, model_func, n=10):
    """Get top-N recommendations for a user from a specific model."""
    # This would need additional parameters based on the model being used
    return []

def display_book_recommendations(user_id, book_ids, books_df, title='Top Recommendations'):
    """Display top book recommendations with details."""
    if len(book_ids) == 0:
        print(f"No recommendations available for user {user_id}")
        return
    
    recommended_books = books_df[books_df['book_id'].isin(book_ids)].copy()
    
    if recommended_books.empty:
        print(f"No matching books found in the dataset for the recommended IDs")
        return
    
    # Format the output for display
    print(f"\n{title} for User {user_id}:")
    for i, (_, book) in enumerate(recommended_books.iterrows(), 1):
        print(f"{i}. '{book['title']}' by {book['authors']}")
        print(f"   Genre: {', '.join(book['genre_list'][:3]) if not isinstance(book['genre_list'], float) else 'Unknown'}")
        print(f"   Average Rating: {book['average_rating']}")
        print(f"   Description: {book['description'][:100]}..." if not pd.isna(book['description']) else "   Description: Not available")
        print()

def get_coverage_metrics(all_book_ids, recommended_book_ids):
    """
    Calculate coverage metrics for a recommender system
    
    Parameters:
    -----------
    all_book_ids : set
        Set of all book IDs in the dataset
    recommended_book_ids : set
        Set of book IDs that were recommended by the model
        
    Returns:
    --------
    dict
        Dictionary containing coverage metrics
    """
    # Calculate catalog coverage (percentage of items that can be recommended)
    catalog_coverage = len(recommended_book_ids) / len(all_book_ids) if len(all_book_ids) > 0 else 0
    
    # Return metrics
    return {
        'Catalog_Coverage': catalog_coverage
    }

# -------------- MAIN EXECUTION ----------------

if __name__ == "__main__":
    print("\nStarting Goodreads Recommender System model evaluation...")
    results = run_all_models()
    print("\nModel evaluation complete!") 