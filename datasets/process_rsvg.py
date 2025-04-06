import os
import torch
import json
from pathlib import Path

def process_rsvg_dataset(data_dir, output_dir):
    """
    Process RSVG dataset into the format expected by TransVG
    
    Args:
        data_dir: Path to the RSVG dataset directory
        output_dir: Path to save the processed dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load train, val, test sets
    train_data = torch.load(os.path.join(data_dir, 'rsvg_train.pth'))
    val_data = torch.load(os.path.join(data_dir, 'rsvg_val.pth'))
    test_data = torch.load(os.path.join(data_dir, 'rsvg_test.pth'))
    
    # Print some debug info
    print(f"Original train data contains {len(train_data)} items")
    if len(train_data) > 0:
        print(f"First item type: {type(train_data[0])}")
        print(f"First item format: {train_data[0]}")
    
    # Process each split
    print("Processing train split...")
    train_processed = process_split(train_data)
    print(f"Processed train data contains {len(train_processed)} items")
    if len(train_processed) > 0:
        print(f"First processed item: {train_processed[0]}")
    
    print("Processing val split...")
    val_processed = process_split(val_data)
    print(f"Processed val data contains {len(val_processed)} items")
    
    print("Processing test split...")
    test_processed = process_split(test_data)
    print(f"Processed test data contains {len(test_processed)} items")
    
    # Save processed data
    torch.save(train_processed, os.path.join(output_dir, 'rsvg_train.pth'))
    torch.save(val_processed, os.path.join(output_dir, 'rsvg_val.pth'))
    torch.save(test_processed, os.path.join(output_dir, 'rsvg_test.pth'))
    
    print(f"Processed dataset saved to {output_dir}")

def process_split(data):
    """
    Process a split of the RSVG dataset into the format expected by TransVG
    
    Args:
        data: RSVG dataset split
        
    Returns:
        List of processed data entries
    """
    processed = []
    
    # Based on the readme and observed data, the items are tuples with format:
    # (image_name, bbox, expression, pos_tags, obj_count)
    
    for item in data:
        try:
            # Direct tuple access since we know the format
            if isinstance(item, tuple) and len(item) >= 3:
                image_name = item[0]
                bbox = item[1]
                expression = item[2]
                
                # Convert bbox to the correct format if needed
                # Check if bbox is already in [x1, y1, x2, y2] format
                if isinstance(bbox, list) and len(bbox) == 4:
                    # TransVG expects a list for bbox
                    bbox_list = bbox
                else:
                    print(f"Unexpected bbox format: {bbox}")
                    continue
                
                # Create the processed item in TransVG format
                # Format: [img_file, None, bbox, phrase, None]
                processed_item = [image_name, None, bbox_list, expression, None]
                processed.append(processed_item)
            else:
                print(f"Skipping item with unexpected format: {type(item)}")
        except Exception as e:
            print(f"Error processing item: {e}")
            print(f"Item: {item}")
    
    return processed

if __name__ == "__main__":
    data_dir = "./rsvg"
    output_dir = "./rsvg/processed"
    process_rsvg_dataset(data_dir, output_dir) 