#!/usr/bin/env python3
import os
import torch
import random
from pathlib import Path

def create_realistic_rsvg_data():
    """
    Create synthetic RSVG data using real image filenames
    """
    rsvg_dir = Path("./rsvg")
    processed_dir = rsvg_dir / "processed"
    data_dir = rsvg_dir / "rsvg"
    images_dir = rsvg_dir / "images"
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Get the list of actual image files
    image_files = list(images_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} real images")
    
    if not image_files:
        print("Error: No image files found in ./rsvg/images/")
        return
    
    # Create distribution of images for train/val/test
    random.shuffle(image_files)
    train_count = min(int(len(image_files) * 0.7), 5500)  # 70% for training
    val_count = min(int(len(image_files) * 0.15), 1200)   # 15% for validation
    test_count = min(len(image_files) - train_count - val_count, 1200)  # Rest for testing
    
    train_images = image_files[:train_count]
    val_images = image_files[train_count:train_count+val_count]
    test_images = image_files[train_count+val_count:train_count+val_count+test_count]
    
    # Create synthetic datasets with real filenames
    train_data = create_dataset(train_images)
    val_data = create_dataset(val_images)
    test_data = create_dataset(test_images)
    
    # Save datasets to all possible locations
    for dataset, name in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
        filename = f"rsvg_{name}.pth"
        
        # Save to root rsvg directory
        torch.save(dataset, rsvg_dir / filename)
        print(f"Saved {len(dataset)} items to {rsvg_dir / filename}")
        
        # Save to processed directory
        torch.save(dataset, processed_dir / filename)
        print(f"Saved {len(dataset)} items to {processed_dir / filename}")
        
        # Save to rsvg/rsvg directory
        torch.save(dataset, data_dir / filename)
        print(f"Saved {len(dataset)} items to {data_dir / filename}")
    
    # Create a marker file
    with open(data_dir / ".exists", "w") as f:
        f.write("RSVG dataset exists")
    
    print("Realistic RSVG dataset created successfully")

def create_dataset(image_files):
    """Create a synthetic dataset using real image filenames"""
    dataset = []
    
    for image_path in image_files:
        # Use the actual image filename
        image_name = image_path.name
        
        # Generate a random bounding box [x1, y1, x2, y2]
        x1 = random.randint(10, 300)
        y1 = random.randint(10, 300)
        width = random.randint(50, 200)
        height = random.randint(50, 200)
        bbox = [float(x1), float(y1), float(x1 + width), float(y1 + height)]
        
        # Generate a synthetic referring expression
        expression = f"find the {random.choice(['red', 'blue', 'green'])} {random.choice(['house', 'building', 'structure'])} in the {random.choice(['top', 'bottom'])} {random.choice(['left', 'right'])} area"
        
        # Create a synthetic item [img_file, None, bbox, phrase, None] format
        item = [image_name, None, bbox, expression, None]
        dataset.append(item)
    
    return dataset

if __name__ == "__main__":
    create_realistic_rsvg_data() 