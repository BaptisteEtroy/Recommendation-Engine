import os
import gzip
import json
import csv

# === CONFIGURATION ===
DATASET_DIR = './Goodreads Dataset'
OUTPUT_DIR = './GoodreadsData1'
SAMPLE_USER_PERCENT = 0.05  # Keep top 50% of users (by review count)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === HELPER FUNCTIONS ===
def safe_truncate(text, max_length):
    """Truncate text safely."""
    if not text:
        return ""
    return text if len(text) <= max_length else text[:max_length] + "..."

def join_field(field):
    """Join list fields into a semicolon separated string."""
    if isinstance(field, list):
        if field and isinstance(field[0], dict):
            return ";".join([",".join([f"{k}:{v}" for k, v in item.items()]) for item in field])
        return ";".join([str(item) for item in field])
    return str(field)

# -------------------------------------------------------------------------
# 1) PROCESS REVIEWS: SAMPLE TOP USERS & MAP IDs
# -------------------------------------------------------------------------
def process_reviews():
    """
    Process reviews from goodreads_reviews_dedup.json.gz.
    First, count reviews per user and select the top SAMPLE_USER_PERCENT users.
    Then, output up to MAX_REVIEWS reviews by these users while mapping old IDs to new sequential IDs.
    Returns:
      kept_users_old: set of original user IDs (strings) that are kept.
      kept_books_old: set of original book IDs (strings) seen in filtered reviews.
      user_id_map: dict mapping old user IDs to new sequential IDs.
      book_id_map: dict mapping old book IDs to new sequential IDs.
    """
    reviews_file = os.path.join(DATASET_DIR, "goodreads_reviews_dedup.json.gz")
    if not os.path.exists(reviews_file):
        print("Reviews file not found!")
        return set(), set(), {}, {}

    # --- PASS 1: Count reviews per user ---
    user_counts = {}
    with gzip.open(reviews_file, 'rt', encoding="utf-8") as infile:
        for line in infile:
            try:
                record = json.loads(line)
            except Exception:
                continue
            user_id = record.get("user_id", "")
            if user_id:
                user_counts[user_id] = user_counts.get(user_id, 0) + 1

    # Sort users by review count descending and select the top SAMPLE_USER_PERCENT.
    all_users_sorted = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)
    num_users_to_keep = int(len(all_users_sorted) * SAMPLE_USER_PERCENT)
    top_users_old = set(user for (user, count) in all_users_sorted[:num_users_to_keep])
    print(f"Pass 1: Found {len(user_counts)} users; keeping top {len(top_users_old)} users based on review count.")

    # --- PASS 2: Write filtered reviews and map IDs ---
    user_id_map = {}
    book_id_map = {}
    next_user_id = 1
    next_book_id = 1

    kept_users_old = set()   # Old user IDs that actually appear in the output.
    kept_books_old = set()   # Old book IDs that appear.
    total_reviews = 0

    reviews_out_path = os.path.join(OUTPUT_DIR, "reviews.csv")
    fieldnames = [
        "review_id", "user_id", "book_id", "rating", "review_text",
        "date_added", "date_updated", "read_at", "started_at", "n_votes", "n_comments"
    ]

    def get_mapped_user_id(old_id):
        nonlocal next_user_id
        if old_id not in user_id_map:
            user_id_map[old_id] = next_user_id
            next_user_id += 1
        return user_id_map[old_id]

    def get_mapped_book_id(old_id):
        nonlocal next_book_id
        if old_id not in book_id_map:
            book_id_map[old_id] = next_book_id
            next_book_id += 1
        return book_id_map[old_id]

    with open(reviews_out_path, "w", newline='', encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        with gzip.open(reviews_file, 'rt', encoding="utf-8") as infile:
            for line in infile:
                try:
                    record = json.loads(line)
                except Exception:
                    continue

                old_user_id = record.get("user_id", "")
                if old_user_id not in top_users_old:
                    continue  # Only keep reviews by top users

                # Map the IDs
                new_user_id = get_mapped_user_id(old_user_id)
                old_book_id = record.get("book_id", "")
                new_book_id = get_mapped_book_id(old_book_id)

                kept_users_old.add(old_user_id)
                kept_books_old.add(old_book_id)

                row = {
                    "review_id": record.get("review_id", ""),
                    "user_id": new_user_id,
                    "book_id": new_book_id,
                    "rating": record.get("rating", ""),
                    "review_text": safe_truncate(record.get("review_text", ""), 1500),
                    "date_added": record.get("date_added", ""),
                    "date_updated": record.get("date_updated", ""),
                    "read_at": record.get("read_at", ""),
                    "started_at": record.get("started_at", ""),
                    "n_votes": record.get("n_votes", ""),
                    "n_comments": record.get("n_comments", "")
                }
                writer.writerow(row)
                total_reviews += 1
#                if total_reviews >= MAX_REVIEWS:
#                    break

    print(f"Processed {total_reviews} reviews into {reviews_out_path}")
    return kept_users_old, kept_books_old, user_id_map, book_id_map

# -------------------------------------------------------------------------
# 2) LOAD GENRE INFORMATION
# -------------------------------------------------------------------------
def load_genres():
    """
    Load genres from goodreads_book_genres_initial.json.gz.
    Returns a dictionary mapping old book_id -> genre string.
    """
    genres_file = os.path.join(DATASET_DIR, "goodreads_book_genres_initial.json.gz")
    genres_dict = {}
    if not os.path.exists(genres_file):
        print("Genres file not found!")
        return genres_dict

    with gzip.open(genres_file, 'rt', encoding="utf-8") as infile:
        for line in infile:
            try:
                record = json.loads(line)
            except Exception:
                continue
            book_id = record.get("book_id", "")
            # Try to extract genre(s) from "genres" or "genre" field
            genre_raw = record.get("genres", record.get("genre", ""))
            genre_str = join_field(genre_raw)
            if book_id:
                genres_dict[book_id] = genre_str
    print(f"Loaded genres for {len(genres_dict)} books from {genres_file}")
    return genres_dict

# -------------------------------------------------------------------------
# 3) PROCESS BOOK METADATA (FILTERED + MERGE GENRES)
# -------------------------------------------------------------------------
def process_books(kept_books_old, book_id_map, genres_dict):
    """
    Process the main book metadata file (goodreads_books.json.gz).
    Only output records for books whose old ID is in kept_books_old.
    Map old book IDs to new sequential IDs using book_id_map.
    Merge genre information from genres_dict (if available).
    Writes output to books.csv.
    """
    books_file = os.path.join(DATASET_DIR, "goodreads_books.json.gz")
    if not os.path.exists(books_file):
        print("Books file not found!")
        return 0

    out_path = os.path.join(OUTPUT_DIR, "books.csv")
    fieldnames = [
        "book_id", "title", "title_without_series", "authors", "description",
        "publisher", "publication_year", "publication_month", "publication_day",
        "edition_information", "isbn", "isbn13", "asin", "kindle_asin",
        "num_pages", "text_reviews_count", "average_rating", "ratings_count",
        "series", "country_code", "language_code", "popular_shelves",
        "similar_books", "format", "link", "url", "image_url", "work_id", "genre"
    ]
    count = 0

    with open(out_path, "w", newline='', encoding="utf-8") as csvfile, \
         gzip.open(books_file, 'rt', encoding="utf-8") as infile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for line in infile:
            try:
                record = json.loads(line)
            except Exception:
                continue

            old_book_id = record.get("book_id", "")
            if old_book_id not in kept_books_old:
                continue

            # Map to new book_id using book_id_map; if missing, skip.
            if old_book_id not in book_id_map:
                continue
            new_book_id = book_id_map[old_book_id]

            # Process authors field as a string
            authors_field = record.get("authors", [])
            if isinstance(authors_field, list) and authors_field:
                if isinstance(authors_field[0], dict):
                    authors_str = ";".join([f"{a.get('author_id','')}:{a.get('name','')}" for a in authors_field])
                else:
                    authors_str = ";".join([str(a) for a in authors_field])
            else:
                authors_str = str(authors_field)

            series_str = join_field(record.get("series", ""))
            shelves_str = join_field(record.get("popular_shelves", ""))
            similar_books_str = join_field(record.get("similar_books", ""))

            # Get genre info from genres_dict (if available)
            genre_str = genres_dict.get(old_book_id, "")

            row = {
                "book_id": new_book_id,
                "title": record.get("title", ""),
                "title_without_series": record.get("title_without_series", ""),
                "authors": authors_str,
                "description": safe_truncate(str(record.get("description", "")), 1500),
                "publisher": record.get("publisher", ""),
                "publication_year": record.get("publication_year", ""),
                "publication_month": record.get("publication_month", ""),
                "publication_day": record.get("publication_day", ""),
                "edition_information": record.get("edition_information", ""),
                "isbn": record.get("isbn", record.get("isbn13", "")),
                "isbn13": record.get("isbn13", ""),
                "asin": record.get("asin", ""),
                "kindle_asin": record.get("kindle_asin", ""),
                "num_pages": record.get("num_pages", ""),
                "text_reviews_count": record.get("text_reviews_count", ""),
                "average_rating": record.get("average_rating", ""),
                "ratings_count": record.get("ratings_count", ""),
                "series": series_str,
                "country_code": record.get("country_code", ""),
                "language_code": record.get("language_code", ""),
                "popular_shelves": shelves_str,
                "similar_books": similar_books_str,
                "format": record.get("format", ""),
                "link": record.get("link", ""),
                "url": record.get("url", ""),
                "image_url": record.get("image_url", ""),
                "work_id": record.get("work_id", ""),
                "genre": genre_str
            }
            writer.writerow(row)
            count += 1

    print(f"Processed {count} book records into {out_path}")
    return count

# -------------------------------------------------------------------------
# 4) PROCESS GENRES: CREATE A SEPARATE CSV FOR GENRE INFO
# -------------------------------------------------------------------------
def process_genres(kept_books_old, book_id_map):
    """
    Extract genre information from goodreads_book_genres_initial.json.gz and write it to genres.csv.
    For each record:
      - If the old book_id is in kept_books_old and has a new mapped ID,
        then write a row with the new book_id and the genre string.
    """
    genres_file = os.path.join(DATASET_DIR, "goodreads_book_genres_initial.json.gz")
    output_csv = os.path.join(OUTPUT_DIR, "genres.csv")
    fieldnames = ["book_id", "genre"]
    count = 0

    if not os.path.exists(genres_file):
        print("Genres file not found!")
        return

    with open(output_csv, "w", newline='', encoding="utf-8") as fout, \
         gzip.open(genres_file, 'rt', encoding="utf-8") as fin:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for line in fin:
            try:
                record = json.loads(line)
            except Exception:
                continue

            old_book_id = record.get("book_id", "")
            # Only process if this old book id is among the kept ones.
            if old_book_id in kept_books_old:
                if old_book_id in book_id_map:
                    new_book_id = book_id_map[old_book_id]
                    genre_raw = record.get("genres", record.get("genre", ""))
                    genre_str = join_field(genre_raw)
                    writer.writerow({"book_id": new_book_id, "genre": genre_str})
                    count += 1

    print(f"Processed {count} genre records into {output_csv}")

# -------------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------------
def main():
    print("Starting extraction of reviews, books, and genre information...")
    kept_users_old, kept_books_old, user_id_map, book_id_map = process_reviews()

    # ✅ Ensure consistency: only keep books that have mappings
    kept_books_old = kept_books_old & set(book_id_map.keys())

    genres_dict = load_genres()
    process_books(kept_books_old, book_id_map, genres_dict)
    process_genres(kept_books_old, book_id_map)

    print("Extraction complete. CSV files (reviews.csv, books.csv, genres.csv) are in:", os.path.abspath(OUTPUT_DIR))

if __name__ == "__main__":
    main()
