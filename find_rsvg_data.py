#!/usr/bin/env python3
import os
import glob
from pathlib import Path

def find_rsvg_data():
    """Search for RSVG data files in the system"""
    # Try to find *.pth files in the rsvg directory and its subdirectories
    rsvg_dir = Path("./rsvg")
    
    print("Searching for potential RSVG data files...")
    pth_files = list(rsvg_dir.glob("**/*.pth"))
    
    for file in pth_files:
        file_size = os.path.getsize(file)
        print(f"Found: {file} (Size: {file_size/1024:.2f} KB)")
    
    # Check if there are any other potential data files
    json_files = list(rsvg_dir.glob("**/*.json"))
    for file in json_files:
        file_size = os.path.getsize(file)
        print(f"Found JSON: {file} (Size: {file_size/1024:.2f} KB)")
    
    txt_files = list(rsvg_dir.glob("**/*.txt"))
    for file in txt_files:
        if file.name != "readme.txt" and file.name != "datasets.txt":
            file_size = os.path.getsize(file)
            print(f"Found TXT: {file} (Size: {file_size/1024:.2f} KB)")
    
    # Check for potential data in parent directories
    parent_dir = rsvg_dir.parent
    print(f"\nChecking parent directory for potential RSVG data: {parent_dir}")
    parent_pth_files = list(parent_dir.glob("**/rsvg*.pth"))
    for file in parent_pth_files:
        if file not in pth_files:
            file_size = os.path.getsize(file)
            print(f"Found: {file} (Size: {file_size/1024:.2f} KB)")

if __name__ == "__main__":
    find_rsvg_data() 