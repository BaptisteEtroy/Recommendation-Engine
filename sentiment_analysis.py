import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from tqdm import tqdm
import pickle

# Constants
SAVED_DATA_DIR = './saved_data'
SENTIMENT_DATA_PATH = os.path.join(SAVED_DATA_DIR, 'sentiment_scores.pkl')
BOOK_SENTIMENT_PATH = os.path.join(SAVED_DATA_DIR, 'book_sentiments.pkl')
USER_SENTIMENT_PATH = os.path.join(SAVED_DATA_DIR, 'user_sentiments.pkl')
os.makedirs(SAVED_DATA_DIR, exist_ok=True)

# BERT model options for sentiment analysis
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"  # A smaller, faster model for sentiment

def load_and_preprocess_reviews(reviews_df):
    """
    Load review data and preprocess it for sentiment analysis
    
    Parameters:
    -----------
    reviews_df : pandas.DataFrame
        DataFrame containing review data with 'review_text' column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with cleaned review text
    """
    print("Preprocessing review text for sentiment analysis...")
    
    # Create a copy to avoid modifying the original
    df = reviews_df.copy()
    
    # Remove rows with missing review text
    df = df[df['review_text'].notna()]
    
    # Truncate very long reviews to 512 tokens (BERT limit)
    df['review_text_processed'] = df['review_text'].str.slice(0, 512)
    
    # Remove reviews that are too short
    df = df[df['review_text_processed'].str.len() > 10]
    
    print(f"Processed {len(df)} reviews for sentiment analysis")
    return df

def initialize_sentiment_model():
    """
    Initialize the BERT model for sentiment analysis
    
    Returns:
    --------
    pipeline
        Hugging Face sentiment analysis pipeline
    """
    print(f"Initializing sentiment model: {MODEL_NAME}")
    
    # Check if CUDA is available
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using {'CUDA' if device == 0 else 'CPU'} for sentiment analysis")
    
    # Initialize the sentiment analysis pipeline
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=device
    )
    
    return sentiment_analyzer

def analyze_sentiments(reviews_df, batch_size=32, save=True):
    """
    Perform sentiment analysis on review texts
    
    Parameters:
    -----------
    reviews_df : pandas.DataFrame
        DataFrame containing preprocessed review texts
    batch_size : int
        Batch size for processing reviews
    save : bool
        Whether to save the sentiment scores
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with original reviews and sentiment scores
    """
    # Check if sentiment scores already exist
    if os.path.exists(SENTIMENT_DATA_PATH):
        print(f"Loading existing sentiment scores from {SENTIMENT_DATA_PATH}")
        sentiment_df = pd.read_pickle(SENTIMENT_DATA_PATH)
        return sentiment_df
    
    print("Performing sentiment analysis on reviews...")
    
    # Preprocess reviews
    processed_df = load_and_preprocess_reviews(reviews_df)
    
    # Initialize sentiment model
    sentiment_analyzer = initialize_sentiment_model()
    
    # Prepare for batched processing
    review_texts = processed_df['review_text_processed'].tolist()
    num_reviews = len(review_texts)
    sentiment_scores = []
    
    # Process in batches
    for i in tqdm(range(0, num_reviews, batch_size), desc="Analyzing sentiment"):
        batch_texts = review_texts[i:min(i + batch_size, num_reviews)]
        try:
            results = sentiment_analyzer(batch_texts)
            
            # Extract sentiment scores (convert to float between -1 and 1)
            for res in results:
                if res['label'] == 'POSITIVE':
                    score = res['score']
                else:
                    score = -res['score']
                sentiment_scores.append(score)
        except Exception as e:
            print(f"Error analyzing batch {i}-{i+batch_size}: {e}")
            # Add neutral scores for this batch
            sentiment_scores.extend([0.0] * len(batch_texts))
    
    # Add sentiment scores to DataFrame
    processed_df['sentiment_score'] = sentiment_scores
    
    # Normalize scores to 0-1 range for easier integration with recommendation models
    processed_df['sentiment_normalized'] = (processed_df['sentiment_score'] + 1) / 2
    
    # Create a DataFrame with only the necessary columns
    result_df = pd.DataFrame({
        'user_id': processed_df['user_id'],
        'book_id': processed_df['book_id'],
        'sentiment_score': processed_df['sentiment_score'],
        'sentiment_normalized': processed_df['sentiment_normalized']
    })
    
    # Save sentiment scores
    if save:
        result_df.to_pickle(SENTIMENT_DATA_PATH)
        print(f"Saved sentiment scores to {SENTIMENT_DATA_PATH}")
    
    return result_df

def get_sentiment_feature(user_id, book_id, sentiment_df):
    """
    Get sentiment feature for a specific user-book pair
    
    Parameters:
    -----------
    user_id : int
        User ID
    book_id : int
        Book ID
    sentiment_df : pandas.DataFrame
        DataFrame containing sentiment scores
        
    Returns:
    --------
    float
        Normalized sentiment score (0-1) or 0.5 if not found
    """
    # Try to find sentiment for this user-book pair
    sentiment = sentiment_df[(sentiment_df['user_id'] == user_id) & 
                             (sentiment_df['book_id'] == book_id)]
    
    if not sentiment.empty:
        return sentiment['sentiment_normalized'].values[0]
    else:
        # Return neutral sentiment if not found
        return 0.5

def analyze_book_sentiment_profiles(sentiment_df):
    """
    Analyze sentiment distributions for each book
    
    Parameters:
    -----------
    sentiment_df : pandas.DataFrame
        DataFrame containing sentiment scores
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with book sentiment profiles
    """
    # Group by book_id and calculate sentiment statistics
    book_sentiments = sentiment_df.groupby('book_id').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'sentiment_normalized': 'mean'
    })
    
    # Flatten the column hierarchy
    book_sentiments.columns = ['sentiment_mean', 'sentiment_std', 'review_count', 'sentiment_norm_mean']
    book_sentiments = book_sentiments.reset_index()
    
    # Fill NaN values with 0 for standard deviation (books with only one review)
    book_sentiments['sentiment_std'] = book_sentiments['sentiment_std'].fillna(0)
    
    # Calculate sentiment polarity - how divisive the book is
    book_sentiments['sentiment_polarity'] = book_sentiments['sentiment_std']
    
    # Save book sentiment profiles
    book_sentiments.to_pickle(BOOK_SENTIMENT_PATH)
    print(f"Saved book sentiment profiles to {BOOK_SENTIMENT_PATH}")
    
    return book_sentiments

def analyze_user_sentiment_profiles(sentiment_df):
    """
    Analyze sentiment patterns for each user
    
    Parameters:
    -----------
    sentiment_df : pandas.DataFrame
        DataFrame containing sentiment scores
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with user sentiment profiles
    """
    # Group by user_id and calculate sentiment statistics
    user_sentiments = sentiment_df.groupby('user_id').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'sentiment_normalized': 'mean'
    })
    
    # Flatten the column hierarchy
    user_sentiments.columns = ['sentiment_mean', 'sentiment_std', 'review_count', 'sentiment_norm_mean']
    user_sentiments = user_sentiments.reset_index()
    
    # Fill NaN values with 0 for standard deviation (users with only one review)
    user_sentiments['sentiment_std'] = user_sentiments['sentiment_std'].fillna(0)
    
    # Calculate sentiment consistency - how consistent a user's sentiment is
    user_sentiments['sentiment_consistency'] = 1 - user_sentiments['sentiment_std'].clip(0, 1)
    
    # Save user sentiment profiles
    user_sentiments.to_pickle(USER_SENTIMENT_PATH)
    print(f"Saved user sentiment profiles to {USER_SENTIMENT_PATH}")
    
    return user_sentiments

# Main function to run the entire sentiment analysis pipeline
def run_sentiment_analysis(reviews_df, sample_size=None):
    """
    Run the complete sentiment analysis pipeline
    
    Parameters:
    -----------
    reviews_df : pandas.DataFrame
        DataFrame containing review data
    sample_size : int, optional
        If provided, only analyze this many reviews as a sample
        
    Returns:
    --------
    tuple
        (sentiment_df, book_sentiments, user_sentiments)
    """
    print("Starting sentiment analysis pipeline...")
    
    # Use a sample if requested (for faster testing)
    if sample_size and len(reviews_df) > sample_size:
        print(f"Using a sample of {sample_size} reviews for sentiment analysis")
        reviews_sample = reviews_df.sample(sample_size, random_state=42)
    else:
        reviews_sample = reviews_df
    
    # Check if all files already exist
    all_files_exist = all([
        os.path.exists(SENTIMENT_DATA_PATH),
        os.path.exists(BOOK_SENTIMENT_PATH),
        os.path.exists(USER_SENTIMENT_PATH)
    ])
    
    if all_files_exist:
        print("All sentiment analysis files already exist, loading them...")
        sentiment_df = pd.read_pickle(SENTIMENT_DATA_PATH)
        book_sentiments = pd.read_pickle(BOOK_SENTIMENT_PATH)
        user_sentiments = pd.read_pickle(USER_SENTIMENT_PATH)
        return sentiment_df, book_sentiments, user_sentiments
    
    # Analyze sentiments for reviews
    sentiment_df = analyze_sentiments(reviews_sample)
    
    # Create book sentiment profiles
    book_sentiments = analyze_book_sentiment_profiles(sentiment_df)
    
    # Create user sentiment profiles
    user_sentiments = analyze_user_sentiment_profiles(sentiment_df)
    
    print("Sentiment analysis complete")
    return sentiment_df, book_sentiments, user_sentiments

# If the file is run directly, perform a test run
if __name__ == "__main__":
    print("Testing sentiment analysis module...")
    
    # Load review data
    DATA_DIR = './GoodreadsData1'
    reviews_file = os.path.join(DATA_DIR, 'reviews.csv')
    
    if os.path.exists(reviews_file):
        reviews_df = pd.read_csv(reviews_file)
        print(f"Loaded {len(reviews_df)} reviews from {reviews_file}")
        
        # Run sentiment analysis on a small sample for testing
        sample_size = min(1000, len(reviews_df))
        print(f"Using a sample of {sample_size} reviews for testing")
        
        sentiment_df, book_sentiments, user_sentiments = run_sentiment_analysis(reviews_df, sample_size=sample_size)
        
        print("\nSample sentiment scores:")
        print(sentiment_df.head())
        
        print("\nSample book sentiment profiles:")
        print(book_sentiments.head())
        
        print("\nSample user sentiment profiles:")
        print(user_sentiments.head())
    else:
        print(f"Review file not found: {reviews_file}") 