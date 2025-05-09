import pandas as pd
import time
from Phishing_detection_with_AI.Datapipeline.Database_connection import supabase

def load_all_features_from_supabase(output_file="url_features.csv"):
    """Loads ALL data from the url_features table in Supabase and saves it to a CSV file."""
    print("[INFO] Loading ALL data from url_features...")

    all_data = []
    page_size = 1000  # Optimal batch size for Supabase
    offset = 0
    retry_count = 0
    max_retries = 3

    try:
        # First, get the total count of records
        count_response = supabase.table("url_features") \
            .select("count", count="exact") \
            .execute()

        total_records = count_response.count
        if not total_records:
            print("[INFO] No records found in url_features.")
            return

        print(f"[INFO] Found {total_records} records. Loading in batches...")

        while offset < total_records:
            while retry_count < max_retries:
                try:
                    response = supabase.table("url_features") \
                        .select("*") \
                        .range(offset, offset + page_size - 1) \
                        .execute()

                    if response.data:
                        all_data.extend(response.data)
                        print(f"[PROGRESS] Loaded {len(all_data)}/{total_records} records")

                    offset += page_size
                    retry_count = 0  # Reset retry counter after success
                    break  # Exit retry loop on success

                except Exception as e:
                    retry_count += 1
                    print(f"[WARNING] Batch {offset}-{offset+page_size} failed (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count >= max_retries:
                        print(f"[ERROR] Failed to load batch {offset}-{offset+page_size} after {max_retries} attempts")
                        raise
                    time.sleep(2 ** retry_count)  # Exponential backoff

        df = pd.DataFrame(all_data)
        df.to_csv("csv_files/all_data.csv", index=False, encoding="utf-8")
        print(f"[SUCCESS] Data saved to {output_file}.")

    except Exception as e:
        print(f"[CRITICAL] Failed to load data: {e}")


def shuffle_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_shuffled.to_csv(output_file, index=False)
    print(f"[SUCCESS] Shuffled CSV saved to {output_file}")

# Usage
shuffle_csv("csv_files/all_data.csv", "csv_files/all_data_shuffled.csv")

