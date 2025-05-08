import os
import csv
import requests
import json
from urllib.parse import urlparse

# Define the paths
csv_path = r"C:\Users\user\OneDrive\Bureau\Academics\pcd\all_data_shuffled.csv"
legitimate_links_dir = r"C:\Users\user\OneDrive\Bureau\Academics\pcd\extract\legitimate_links16"
phishing_links_dir = r"C:\Users\user\OneDrive\Bureau\Academics\pcd\extract\phishing_links6"
last_processed_line_path = r"C:\Users\user\OneDrive\Bureau\Academics\pcd\extract\last_processed_line.txt"

# Ensure the output directories exist
os.makedirs(legitimate_links_dir, exist_ok=True)
os.makedirs(phishing_links_dir, exist_ok=True)

def save_to_json(content, file_path):
    """Save content to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(content, file, indent=4)

def fetch_url_content(session, url):
    """Fetch the content of a URL using a session."""
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None

def get_last_processed_line():
    """Get the last processed line from the file."""
    if os.path.exists(last_processed_line_path):
        with open(last_processed_line_path, 'r', encoding='utf-8') as file:
            return int(file.read().strip())
    return 0

def save_last_processed_line(line_number):
    """Save the last processed line to the file."""
    with open(last_processed_line_path, 'w', encoding='utf-8') as file:
        file.write(str(line_number))

def get_next_file_number(directory):
    """Get the next file number for the directory."""
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    return len(files) + 1

def main():
    last_processed_line = get_last_processed_line()
    current_line = 0

    # Create a session to handle consecutive requests
    with requests.Session() as session:
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                current_line += 1
                if current_line <= last_processed_line:
                    continue

                url = row['url']
                label = int(row['target'])

                # Fetch the content of the URL
                content = fetch_url_content(session, url)
                if content is None:
                    continue

                # Determine the output directory
                if label == 0:
                    output_dir = legitimate_links_dir
                else:
                    output_dir = phishing_links_dir

                # Get the next file number for the directory
                file_number = get_next_file_number(output_dir)
                file_name = f"{file_number}.json"
                file_path = os.path.join(output_dir, file_name)

                # Save the content to a JSON file
                save_to_json({'url': url, 'content': content}, file_path)
                print(f"Saved {url} to {file_path}")

                # Save the last processed line
                save_last_processed_line(current_line)

if __name__ == "__main__":
    main()






























