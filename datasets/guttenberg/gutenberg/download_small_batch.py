#!/usr/bin/env python3
"""
Simple Gutenberg Downloader - Downloads books to reach target size
Uses correct URL format: https://www.gutenberg.org/files/{id}/{id}.txt
"""

import os
import subprocess
import sys

# Books to download (PG ID : title)
# This list should give us ~100MB total of classic literature
BOOKS_TO_DOWNLOAD = [
    ("6431", "The Count of Monte Cristo - Dumas"),
    ("20203", "On the Origin of Species - Darwin"),
    ("1342", "Pride and Prejudice - Austen"),
    ("1661", "Moby Dick - Melville"),
    ("1184", "Great Expectations - Dickens"),
    ("1080", "Uncle Tom's Cabin - Stowe"),
    ("1260", "Tom Sawyer - Twain"),
    ("74", "Huckleberry Finn - Twain"),
    ("84", "Frankenstein - Shelley"),
    ("2554", "Sherlock Holmes - Doyle"),
    ("345", "Dr. Jekyll and Mr. Hyde"),
    ("1352", "A Tale of Two Cities - Dickens"),
    ("1400", "King James Bible"),
    ("12947", "The Iliad - Homer"),
    ("12948", "The Odyssey - Homer"),
    ("5200", "War and Peace - Tolstoy"),
    ("2701", "Leaves of Grass - Whitman"),
]

BASE_URL = "https://www.gutenberg.org/files"

def download_book(pg_id, title):
    """Download a single book from Gutenberg"""
    for url_suffix in [f"{pg_id}.txt", f"{pg_id}-0.txt"]:
        url = f"{BASE_URL}/{pg_id}/{url_suffix}"
        output_file = f"data/raw/PG{pg_id}_raw.txt"
        
        if os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
            return os.path.getsize(output_file)
        
        try:
            result = subprocess.run(
                ["wget", "-q", "--timeout=30", "-O", output_file, url],
                capture_output=True, timeout=60
            )
            if result.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
                size = os.path.getsize(output_file)
                return size
        except:
            pass
    
    # Clean up failed download
    output_file = f"data/raw/PG{pg_id}_raw.txt"
    if os.path.exists(output_file):
        os.remove(output_file)
    return 0

def main():
    import argparse
    parser = argparse.ArgumentParser("Download Gutenberg books to reach target size")
    parser.add_argument("-s", "--size", default=100, type=int, help="Target size in MB")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    
    target_bytes = args.size * 1024 * 1024
    os.makedirs("data/raw", exist_ok=True)
    
    print(f"Target: {args.size}MB")
    print("-" * 50)
    
    total_size = 0
    downloaded = 0
    
    for pg_id, title in BOOKS_TO_DOWNLOAD:
        if total_size >= target_bytes:
            break
        size = download_book(pg_id, title)
        if size > 0:
            total_size += size
            downloaded += 1
            if not args.quiet:
                print(f"  [+] {title}: {size/(1024*1024):.2f}MB")
        else:
            if not args.quiet:
                print(f"  [-] {title}: FAILED")
    
    print("-" * 50)
    print(f"Downloaded: {downloaded} books, {total_size/(1024*1024):.2f}MB")
    print(f"Target: {args.size}MB")
    
    if total_size < target_bytes:
        print(f"Need {((target_bytes-total_size)/(1024*1024)):.2f}MB more")

if __name__ == '__main__':
    sys.exit(main())