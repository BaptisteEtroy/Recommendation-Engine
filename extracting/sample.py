import os
import csv

# === CONFIGURATION ===
DATASET_DIR = './Filtered_Dataset'  # Existing folder with CSVs
OUTPUT_DIR = './Filtered_Dataset_ReviewsOnly'  # New folder to create
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # 1) Read reviews.csv to find all user_ids and book_ids
    reviews_path = os.path.join(DATASET_DIR, 'reviews.csv')
    kept_users = set()
    kept_books = set()

    if not os.path.exists(reviews_path):
        print(f"ERROR: {reviews_path} not found.")
        return

    with open(reviews_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = row.get('user_id', '').strip()
            book_id = row.get('book_id', '').strip()
            if user_id:
                kept_users.add(user_id)
            if book_id:
                kept_books.add(book_id)

    print(f"Found {len(kept_users)} unique users and {len(kept_books)} unique books in reviews.csv.")

    # 2) Filter interactions.csv
    interactions_in = os.path.join(DATASET_DIR, 'interactions.csv')
    interactions_out = os.path.join(OUTPUT_DIR, 'interactions.csv')
    filter_interactions(interactions_in, interactions_out, kept_users, kept_books)

    # 3) Filter books.csv
    books_in = os.path.join(DATASET_DIR, 'books.csv')
    books_out = os.path.join(OUTPUT_DIR, 'books.csv')
    filter_books(books_in, books_out, kept_books)

    # 4) Filter authors.csv
    authors_in = os.path.join(DATASET_DIR, 'authors.csv')
    authors_out = os.path.join(OUTPUT_DIR, 'authors.csv')
    filter_authors(authors_in, authors_out, kept_books)

    # 5) Filter users.csv
    users_in = os.path.join(DATASET_DIR, 'users.csv')
    users_out = os.path.join(OUTPUT_DIR, 'users.csv')
    filter_users(users_in, users_out, kept_users)

    # 6) Filter (or copy) reviews.csv
    reviews_out = os.path.join(OUTPUT_DIR, 'reviews.csv')
    filter_reviews(reviews_path, reviews_out, kept_users, kept_books)

    print("\n--- Done. Filtered files are in", os.path.abspath(OUTPUT_DIR))


def filter_interactions(input_csv, output_csv, kept_users, kept_books):
    """Keep only rows whose user_id is in kept_users and book_id is in kept_books."""
    if not os.path.exists(input_csv):
        print(f"Skipping interactions: {input_csv} not found.")
        return
    count_in, count_out = 0, 0
    with open(input_csv, 'r', encoding='utf-8') as fin, \
         open(output_csv, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(fout, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for row in reader:
            count_in += 1
            u = row.get('user_id', '').strip()
            b = row.get('book_id', '').strip()
            if (u in kept_users) and (b in kept_books):
                writer.writerow(row)
                count_out += 1

    print(f"Filtered interactions.csv: {count_in} -> {count_out} rows, saved to {output_csv}")


def filter_books(input_csv, output_csv, kept_books):
    """Keep only rows whose book_id is in kept_books."""
    if not os.path.exists(input_csv):
        print(f"Skipping books: {input_csv} not found.")
        return
    count_in, count_out = 0, 0
    with open(input_csv, 'r', encoding='utf-8') as fin, \
         open(output_csv, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(fout, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for row in reader:
            count_in += 1
            b = row.get('book_id', '').strip()
            if b in kept_books:
                writer.writerow(row)
                count_out += 1

    print(f"Filtered books.csv: {count_in} -> {count_out} rows, saved to {output_csv}")


def filter_authors(input_csv, output_csv, kept_books):
    """
    Keep only rows for authors who have at least one book in kept_books.
    The 'books' column is assumed to be semicolon-separated book IDs.
    We only keep the subset of those book IDs that are in kept_books.
    If none remain, we skip that author row.
    """
    if not os.path.exists(input_csv):
        print(f"Skipping authors: {input_csv} not found.")
        return
    count_in, count_out = 0, 0
    with open(input_csv, 'r', encoding='utf-8') as fin, \
         open(output_csv, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(fout, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for row in reader:
            count_in += 1
            old_books_str = row.get('books', '').strip()
            if not old_books_str:
                continue

            old_books_list = old_books_str.split(';')
            # Filter out only books in kept_books
            filtered_list = [b for b in old_books_list if b.strip() in kept_books]
            if len(filtered_list) == 0:
                # No relevant books left, skip
                continue

            # Join them back
            row['books'] = ";".join(filtered_list)
            writer.writerow(row)
            count_out += 1

    print(f"Filtered authors.csv: {count_in} -> {count_out} rows, saved to {output_csv}")


def filter_users(input_csv, output_csv, kept_users):
    """Keep only rows whose user_id is in kept_users."""
    if not os.path.exists(input_csv):
        print(f"Skipping users: {input_csv} not found.")
        return
    count_in, count_out = 0, 0
    with open(input_csv, 'r', encoding='utf-8') as fin, \
         open(output_csv, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(fout, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for row in reader:
            count_in += 1
            u = row.get('user_id', '').strip()
            if u in kept_users:
                writer.writerow(row)
                count_out += 1

    print(f"Filtered users.csv: {count_in} -> {count_out} rows, saved to {output_csv}")


def filter_reviews(input_csv, output_csv, kept_users, kept_books):
    """Keep only rows whose user_id is in kept_users and book_id is in kept_books."""
    if not os.path.exists(input_csv):
        print(f"Skipping reviews: {input_csv} not found.")
        return
    count_in, count_out = 0, 0
    with open(input_csv, 'r', encoding='utf-8') as fin, \
         open(output_csv, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(fout, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for row in reader:
            count_in += 1
            u = row.get('user_id', '').strip()
            b = row.get('book_id', '').strip()
            if (u in kept_users) and (b in kept_books):
                writer.writerow(row)
                count_out += 1

    print(f"Filtered reviews.csv: {count_in} -> {count_out} rows, saved to {output_csv}")


if __name__ == "__main__":
    main()
