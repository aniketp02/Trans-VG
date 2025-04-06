#!/usr/bin/env python3
import os
import torch
import json
from pathlib import Path

def check_rsvg_files():
    """Check the RSVG dataset files and their content"""
    rsvg_dir = Path("./rsvg")
    
    # List possible file locations
    possible_paths = [
        rsvg_dir / "rsvg_train.pth",
        rsvg_dir / "rsvg_val.pth",
        rsvg_dir / "rsvg_test.pth",
        rsvg_dir / "rsvg" / "rsvg_train.pth",
        rsvg_dir / "rsvg" / "rsvg_val.pth",
        rsvg_dir / "rsvg" / "rsvg_test.pth",
        rsvg_dir / "processed" / "rsvg_train.pth",
        rsvg_dir / "processed" / "rsvg_val.pth",
        rsvg_dir / "processed" / "rsvg_test.pth",
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"File exists: {path}")
            try:
                data = torch.load(str(path))
                if hasattr(data, "__len__"):
                    print(f"  Contains {len(data)} items")
                    if len(data) > 0:
                        print(f"  First item: {data[0]}")
                        print(f"  Type: {type(data[0])}")
                else:
                    print(f"  Data type: {type(data)}")
            except Exception as e:
                print(f"  Error loading file: {e}")
        else:
            print(f"File does not exist: {path}")
    
    # Check if the images directory exists and contains images
    images_dir = rsvg_dir / "images"
    if images_dir.exists():
        print(f"Images directory exists: {images_dir}")
        image_files = list(images_dir.glob("*.jpg"))
        if image_files:
            print(f"  Contains {len(image_files)} images")
            print(f"  Sample images: {image_files[:5]}")
        else:
            print("  No images found in the directory")
    else:
        print(f"Images directory does not exist: {images_dir}")
    
    # Check the readme file for guidance
    readme_path = rsvg_dir / "readme.txt"
    if readme_path.exists():
        print(f"Readme file exists: {readme_path}")
        with open(readme_path, 'r') as f:
            print(f"Readme content:\n{f.read()}")
    else:
        print(f"Readme file does not exist: {readme_path}")

if __name__ == "__main__":
    check_rsvg_files() 