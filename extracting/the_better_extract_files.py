import os
import gzip
import json
import csv
import pandas as pd

# === CONFIGURATION ===
DATASET_DIR = './Goodreads Dataset'
OUTPUT_DIR = './Filtered_Dataset'
# Increased record limits to push total output closer to 3GB.
MAX_BOOKS = 400000         # maximum number of book records
MAX_AUTHORS = 200000        # maximum number of unique author records
MAX_REVIEWS = 2000000      # maximum number of review records
MAX_INTERACTIONS = 4000000 # maximum number of interaction records

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === HELPER FUNCTIONS ===
def safe_truncate(text, max_length):
    """Truncate a string safely."""
    if not text:
        return ""
    return text if len(text) <= max_length else text[:max_length] + "..."

def join_field(field):
    """Join list fields into a semicolon separated string."""
    if isinstance(field, list):
        # If list items are dicts, join key:value pairs for each item.
        if field and isinstance(field[0], dict):
            return ";".join([",".join([f"{k}:{v}" for k, v in item.items()]) for item in field])
        return ";".join([str(item) for item in field])
    return str(field)

# === PROCESS BOOK METADATA ===
def process_books():
    """
    Process the main book metadata file (goodreads_books.json.gz).
    Unpacks nearly every field available.
    Writes a CSV file with expanded metadata.
    """
    books_file = os.path.join(DATASET_DIR, "goodreads_books.json.gz")
    if not os.path.exists(books_file):
        print("Books file not found!")
        return 0

    out_path = os.path.join(OUTPUT_DIR, "books.csv")
    # Include as many fields as available in the dataset.
    fieldnames = [
        "book_id", "title", "title_without_series", "authors", "description",
        "publisher", "publication_year", "publication_month", "publication_day",
        "edition_information", "isbn", "isbn13", "asin", "kindle_asin",
        "num_pages", "text_reviews_count", "average_rating", "ratings_count",
        "series", "country_code", "language_code", "popular_shelves",
        "similar_books", "format", "link", "url", "image_url", "work_id"
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

            # Process the authors field.
            authors_field = record.get("authors", [])
            if isinstance(authors_field, list) and authors_field:
                if isinstance(authors_field[0], dict):
                    authors_str = ";".join([f"{a.get('author_id','')}:{a.get('name','')}" for a in authors_field])
                else:
                    authors_str = ";".join([str(a) for a in authors_field])
            else:
                authors_str = str(authors_field)

            # Process list fields using join_field helper.
            series_str = join_field(record.get("series", ""))
            shelves_str = join_field(record.get("popular_shelves", ""))
            similar_books_str = join_field(record.get("similar_books", ""))

            row = {
                "book_id": record.get("book_id", ""),
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
                "work_id": record.get("work_id", "")
            }
            writer.writerow(row)
            count += 1
            if count >= MAX_BOOKS:
                break
    print(f"Processed {count} book records into {out_path}")
    return os.path.getsize(out_path)

# === PROCESS AUTHOR METADATA ===
def process_authors():
    """
    Process the author metadata file (goodreads_book_authors.json.gz).
    Extracts unique author records with as much metadata as possible.
    Writes a CSV file with one record per unique author.
    """
    authors_file = None
    for fname in os.listdir(DATASET_DIR):
        if "book_authors" in fname.lower() and fname.endswith(".json.gz"):
            authors_file = os.path.join(DATASET_DIR, fname)
            break
    if not authors_file:
        print("Authors file not found!")
        return 0

    out_path = os.path.join(OUTPUT_DIR, "authors.csv")
    fieldnames = [
        "author_id", "name", "role", "books"
    ]
    seen_authors = {}
    count = 0
    with open(out_path, "w", newline='', encoding="utf-8") as csvfile, \
         gzip.open(authors_file, 'rt', encoding="utf-8") as infile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for line in infile:
            try:
                record = json.loads(line)
            except Exception:
                continue
            author_id = record.get("author_id", "")
            if not author_id:
                continue
            if author_id not in seen_authors:
                seen_authors[author_id] = {
                    "author_id": author_id,
                    "name": record.get("name", ""),
                    "role": record.get("role", ""),
                    "books": record.get("book_id", "")
                }
                count += 1
                if count >= MAX_AUTHORS:
                    break
            else:
                existing = seen_authors[author_id]
                book_id = record.get("book_id", "")
                if book_id and book_id not in existing["books"]:
                    existing["books"] += f";{book_id}"
        for author in seen_authors.values():
            writer.writerow(author)
    print(f"Processed {len(seen_authors)} unique author records into {out_path}")
    return os.path.getsize(out_path)

# === PROCESS REVIEWS ===
def process_reviews():
    """
    Process review files (all .json.gz files with 'review' in the name).
    Filters out reviews with rating 0 and keeps extensive metadata.
    Writes a CSV file with selected review fields.
    """
    review_files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)
                    if "review" in f.lower() and f.endswith(".json.gz")]
    if not review_files:
        print("No review files found!")
        return 0

    out_path = os.path.join(OUTPUT_DIR, "reviews.csv")
    fieldnames = [
        "review_id", "user_id", "book_id", "rating", "review_text",
        "date_added", "date_updated", "read_at", "started_at", "n_votes", "n_comments"
    ]
    total_count = 0
    with open(out_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for rfile in review_files:
            with gzip.open(rfile, 'rt', encoding="utf-8") as infile:
                for line in infile:
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    rating = record.get("rating", 0)
                    if rating == 0:
                        continue
                    row = {
                        "review_id": record.get("review_id", ""),
                        "user_id": record.get("user_id", ""),
                        "book_id": record.get("book_id", ""),
                        "rating": rating,
                        "review_text": safe_truncate(record.get("review_text", ""), 1500),
                        "date_added": record.get("date_added", ""),
                        "date_updated": record.get("date_updated", ""),
                        "read_at": record.get("read_at", ""),
                        "started_at": record.get("started_at", ""),
                        "n_votes": record.get("n_votes", ""),
                        "n_comments": record.get("n_comments", "")
                    }
                    writer.writerow(row)
                    total_count += 1
                    if total_count >= MAX_REVIEWS:
                        break
            print(f"Finished processing reviews from {os.path.basename(rfile)}")
            if total_count >= MAX_REVIEWS:
                break
    print(f"Processed {total_count} review records into {out_path}")
    return os.path.getsize(out_path)

# === PROCESS INTERACTIONS ===
def process_interactions():
    """
    Process interactions from both CSV and JSON sources.
    Filters out records with a rating of 0.
    Writes a combined interactions.csv file.
    """
    out_path = os.path.join(OUTPUT_DIR, "interactions.csv")
    fieldnames = [
        "user_id", "book_id", "rating", "is_read", "is_reviewed",
        "date_added", "date_updated", "read_at", "started_at"
    ]
    total_count = 0
    with open(out_path, "w", newline='', encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        
        # Process CSV-based interactions first.
        for fname in os.listdir(DATASET_DIR):
            if "interaction" in fname.lower() and fname.endswith(".csv"):
                csv_path = os.path.join(DATASET_DIR, fname)
                with open(csv_path, "r", encoding="utf-8") as infile:
                    reader = csv.DictReader(infile)
                    for row in reader:
                        try:
                            rating = float(row.get("rating", 0))
                        except ValueError:
                            rating = 0
                        if rating == 0:
                            continue
                        writer.writerow(row)
                        total_count += 1
                        if total_count >= MAX_INTERACTIONS:
                            break
                print(f"Processed interactions from {fname}")
                if total_count >= MAX_INTERACTIONS:
                    break

        # Process JSON-based interactions.
        if total_count < MAX_INTERACTIONS:
            for fname in os.listdir(DATASET_DIR):
                if "interaction" in fname.lower() and fname.endswith(".json.gz"):
                    json_path = os.path.join(DATASET_DIR, fname)
                    with gzip.open(json_path, 'rt', encoding="utf-8") as infile:
                        for line in infile:
                            try:
                                record = json.loads(line)
                            except Exception:
                                continue
                            if record.get("rating", 0) == 0:
                                continue
                            row = {
                                "user_id": record.get("user_id", ""),
                                "book_id": record.get("book_id", ""),
                                "rating": record.get("rating", ""),
                                "is_read": record.get("is_read", ""),
                                "is_reviewed": record.get("is_reviewed", ""),
                                "date_added": record.get("date_added", ""),
                                "date_updated": record.get("date_updated", ""),
                                "read_at": record.get("read_at", ""),
                                "started_at": record.get("started_at", "")
                            }
                            writer.writerow(row)
                            total_count += 1
                            if total_count >= MAX_INTERACTIONS:
                                break
                    print(f"Processed interactions from {fname}")
                    if total_count >= MAX_INTERACTIONS:
                        break
    print(f"Processed {total_count} interaction records into {out_path}")
    return os.path.getsize(out_path)

# === PROCESS USER DATA ===
def process_users():
    """
    Process user metadata.
    Here we assume the available user file is a CSV mapping file.
    """
    users_file = None
    for fname in os.listdir(DATASET_DIR):
        if "user" in fname.lower() and fname.endswith(".csv"):
            users_file = os.path.join(DATASET_DIR, fname)
            break
    if not users_file:
        print("User file not found!")
        return 0

    out_path = os.path.join(OUTPUT_DIR, "users.csv")
    try:
        df = pd.read_csv(users_file)
        df.to_csv(out_path, index=False)
        print(f"Processed users from {os.path.basename(users_file)} into {out_path}")
    except Exception as e:
        print(f"Error processing users: {e}")
        return 0
    return os.path.getsize(out_path)

# === MAIN FUNCTION ===
def main():
    print("Starting extraction and filtering from the Goodreads Dataset...")
    total_size = 0

    size_books = process_books()
    total_size += size_books

    size_authors = process_authors()
    total_size += size_authors

    size_reviews = process_reviews()
    total_size += size_reviews

    size_interactions = process_interactions()
    total_size += size_interactions

    size_users = process_users()
    total_size += size_users

    print("\n--- Extraction Complete ---")
    print(f"Total output size: {total_size / (1024**3):.2f} GB")
    print("CSV files created in:", os.path.abspath(OUTPUT_DIR))
    if total_size > 3 * 1024**3:
        print("Warning: Output exceeds 3GB. Consider reducing record limits (MAX_*) if needed.")

if __name__ == "__main__":
    main()
