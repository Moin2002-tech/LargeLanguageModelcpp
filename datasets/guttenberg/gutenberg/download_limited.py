#!/usr/bin/env python3
"""
Limited Gutenberg Downloader
Downloads books until reaching a target size (default: 200MB)
"""

import os
import subprocess
import sys
import time
from src.utils import populate_raw_from_mirror, list_duplicates_in_mirror

# Target size in bytes (200MB)
TARGET_SIZE = 200 * 1024 * 1024  # 200 MB

def get_raw_size(raw_dir):
    """Calculate total size of files in raw directory"""
    total = 0
    if not os.path.isdir(raw_dir):
        return 0
    for dirpath, dirnames, filenames in os.walk(raw_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp) and not fp.endswith('.dummy'):
                total += os.path.getsize(fp)
    return total

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        "Download Gutenberg books up to a target size"
    )
    parser.add_argument(
        "-m", "--mirror",
        help="Path to mirror folder",
        default='data/.mirror/',
        type=str
    )
    parser.add_argument(
        "-r", "--raw",
        help="Path to raw folder",
        default='data/raw/',
        type=str
    )
    parser.add_argument(
        "-M", "--metadata",
        help="Path to metadata folder",
        default='metadata/',
        type=str
    )
    parser.add_argument(
        "-s", "--size",
        help="Target size in MB (default: 200)",
        default=200,
        type=int
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode"
    )
    parser.add_argument(
        "-owr", "--overwrite_raw",
        action="store_true",
        help="Overwrite files in raw"
    )
    
    args = parser.parse_args()
    
    target_bytes = args.size * 1024 * 1024
    
    # Ensure directories exist
    for d in [args.mirror, args.raw, args.metadata]:
        os.makedirs(d, exist_ok=True)
    
    print(f"Target size: {args.size}MB ({target_bytes} bytes)")
    print(f"Initial raw size: {get_raw_size(args.raw) / (1024*1024):.2f}MB")
    
    if not args.quiet:
        print("Starting download automatically (no interactive mode in background)...")
    # Skip interactive prompt for background runs
    
    # Step 1: rsync to mirror (this downloads the index of all books)
    print("\n[1/3] Syncing with Gutenberg mirror via rsync...")
    vstring = "" if args.quiet else "v"
    
    rsync_cmd = [
        "rsync", "-am%s" % vstring,
        "--include", "*/",
        "--include", "[p123456789][g0123456789]*[.-][t0][x.]t[x.]*[t8]",
        "--exclude", "*",
        "gutenberg.pglaf.org::gutenberg",
        args.mirror
    ]
    
    result = subprocess.run(rsync_cmd)
    if result.returncode != 0:
        print("rsync failed!")
        sys.exit(1)
    
    # Step 2: Get duplicates list
    print("[2/3] Processing mirror...")
    dups_list = list_duplicates_in_mirror(mirror_dir=args.mirror)
    
    # Step 3: Populate raw with size limit
    print("[3/3] Populating raw directory (stopping at target size)...")
    
    # We need to manually control the population to limit size
    # Since populate_raw_from_mirror doesn't support size limits, we'll do it ourselves
    
    import glob
    
    current_size = get_raw_size(args.raw)
    books_downloaded = 0
    
    for dirName, subdirList, fileList in os.walk(args.mirror):
        for matchpath in glob.iglob(os.path.join(dirName, "[p123456789][g0123456789][0-9]*")):
            fname = matchpath.split("/")[-1]
            
            if matchpath not in dups_list:
                # Check file pattern
                if (len(fname.split(".")) == 2 and len(fname.split("-")) == 2 and fname[-6:] == "-0.txt") \
                   or (len(fname.split(".")) == 3 and len(fname.split("-")) == 1 and fname[-9:] == ".txt.utf8"):
                    
                    # Get PG number
                    PGnumber = fname.replace("-0.txt", "").replace(".txt.utf8", "").replace("pg", "")
                    
                    if not PGnumber.isnumeric():
                        continue
                    
                    source = os.path.join(dirName, fname)
                    target = os.path.join(args.raw, f"PG{PGnumber}_raw.txt")
                    
                    if (not os.path.isfile(target)) or args.overwrite_raw:
                        # Check if adding this file would exceed target
                        file_size = os.path.getsize(source)
                        
                        if current_size + file_size > target_bytes and books_downloaded > 0:
                            if not args.quiet:
                                print(f"\nTarget size reached after downloading {books_downloaded} books")
                                print(f"Current size: {current_size / (1024*1024):.2f}MB / Target: {args.size}MB")
                            return
                        
                        # Hard link the file
                        subprocess.call(["ln", "-f", source, target])
                        current_size += file_size
                        books_downloaded += 1
                        
                        if not args.quiet and books_downloaded % 10 == 0:
                            print(f"Downloaded {books_downloaded} books... ({current_size / (1024*1024):.2f}MB)")
    
    print(f"\nDone! Downloaded {books_downloaded} books")
    print(f"Total size: {current_size / (1024*1024):.2f}MB")

if __name__ == '__main__':
    main()
