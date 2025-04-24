import os
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Callable

from .transforms import get_transform
from .mask_transforms import prepare_mask, apply_mask_transforms, resize_binary_mask

class RSVGSegmentationDataset(Dataset):
    """
    Dataset for RSVG segmentation data, loading both images and binary masks
    
    Args:
        root_dir: Root directory of the RSVG dataset
        split: Dataset split ('train', 'val', or 'test')
        transform: Optional transforms to apply to images and masks
        img_size: Target image size (default: 224)
        load_mask: Whether to load segmentation masks (default: True)
        aug_intensity: Augmentation intensity ('none', 'light', 'medium', 'heavy')
        mask_dir: Directory containing segmentation masks (default: 'Masks')
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        img_size: int = 224,
        load_mask: bool = True,
        aug_intensity: str = 'medium',
        mask_dir: str = 'Masks'
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.load_mask = load_mask
        self.mask_dir = mask_dir
        
        # Set up image directories
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(root_dir, mask_dir)
        self.ann_dir = os.path.join(root_dir, 'Annotations')
        
        # Get split file path
        split_file = os.path.join(root_dir, f'{split}.txt')
        
        # Load image IDs from split file
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.image_ids = [line.strip() for line in f.readlines()]
        else:
            # Fallback to torch.load if split file is not txt
            data = torch.load(os.path.join(root_dir, f'{split}.pt'), weights_only=True)
            self.image_ids = [entry[0] for entry in data]  # First element is image name
        
        # Set up transforms
        if transform is None:
            self.transform = get_transform(
                img_size=img_size,
                is_training=(split == 'train'),
                use_augmentation=True,
                aug_intensity=aug_intensity
            )
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def format_image_id(self, img_id: str) -> str:
        """
        Format image ID to match the annotation file naming convention.
        The annotation files are named with 6-digit zero-padded numbers.
        """
        try:
            # Format with 6 digits (zero-padded)
            return f"{int(img_id):05d}"
        except ValueError:
            # If conversion to int fails, return as is
            return img_id
    
    def load_annotation(self, img_id: str) -> Tuple[str, List[Tuple[str, Tuple[int, int, int, int]]]]:
        """
        Load XML annotation for an image ID
        
        Returns:
            Tuple of (filename, [(class_name, bbox)])
        """
        formatted_id = self.format_image_id(img_id)
        xml_path = os.path.join(self.ann_dir, f'{formatted_id}.xml')
        
        if not os.path.exists(xml_path):
            print(f"XML file not found: {xml_path}")
            return None, []
            
        try:
            # Read the XML file
            with open(xml_path, 'r') as f:
                xml_content = f.read()
            
            # Parse the XML
            root = ET.fromstring(xml_content)
            
            # Get image filename
            filename = root.find('filename').text
            
            # Extract objects
            objects = []
            for obj in root.findall('object'):
                # Get class name
                class_name = None
                for child in obj:
                    if child.tag in ['n', 'name']:
                        if child.text and child.text.strip():
                            class_name = child.text.strip()
                            break
                
                if not class_name:
                    continue
                    
                # Get bounding box
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                    
                try:
                    xmin = int(float(bbox.find('xmin').text))
                    ymin = int(float(bbox.find('ymin').text))
                    xmax = int(float(bbox.find('xmax').text))
                    ymax = int(float(bbox.find('ymax').text))
                    
                    # Store class and bbox
                    objects.append((class_name, (xmin, ymin, xmax, ymax)))
                except (ValueError, AttributeError):
                    continue
            
            return filename, objects
        except Exception as e:
            print(f"Error parsing annotation for {img_id}: {e}")
            return None, []
    
    def load_mask(self, img_id: str, img_shape: Tuple[int, int]) -> np.ndarray:
        """
        Load mask for an image
        
        Args:
            img_id: Image ID
            img_shape: Shape of the image (height, width)
        
        Returns:
            Binary mask as numpy array (H x W)
        """
        height, width = img_shape
        formatted_id = self.format_image_id(img_id)
        
        # Try loading mask from mask directory
        mask_path = os.path.join(self.mask_dir, f'{formatted_id}.png')
        
        if os.path.exists(mask_path):
            # Load mask directly
            mask = np.array(Image.open(mask_path).convert('L'))
            mask = (mask > 0).astype(np.float32)
            return mask
        
        # If mask file doesn't exist, try to generate from annotations
        _, objects = self.load_annotation(img_id)
        
        if not objects:
            # If no objects found, return empty mask
            return np.zeros((height, width), dtype=np.float32)
        
        # Create mask from bounding boxes
        mask = np.zeros((height, width), dtype=np.float32)
        
        for _, bbox in objects:
            xmin, ymin, xmax, ymax = bbox
            # Ensure coordinates are within image bounds
            xmin = max(0, min(xmin, width - 1))
            ymin = max(0, min(ymin, height - 1))
            xmax = max(0, min(xmax, width - 1))
            ymax = max(0, min(ymax, height - 1))
            
            # Fill the mask region
            mask[ymin:ymax+1, xmin:xmax+1] = 1.0
        
        return mask
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_id = self.image_ids[idx]
        formatted_id = self.format_image_id(img_id)
        
        # Load image
        filename, objects = self.load_annotation(img_id)
        
        if filename is None:
            # If annotation not found, use image ID as filename
            img_path = os.path.join(self.img_dir, f'{formatted_id}.jpg')
            
            # Also try 5-digit format (annotations use 6 digits but images might use 5)
            try:
                five_digit_id = f"{int(img_id):05d}"
                alt_img_path = os.path.join(self.img_dir, f'{five_digit_id}.jpg')
                if os.path.exists(alt_img_path):
                    img_path = alt_img_path
            except ValueError:
                pass
        else:
            img_path = os.path.join(self.img_dir, filename)
        
        # Try different file extensions if file not found
        if not os.path.exists(img_path):
            # Try 5-digit format with different extensions
            try:
                five_digit_id = f"{int(img_id):05d}"
                for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                    alt_path = os.path.join(self.img_dir, f'{five_digit_id}{ext}')
                    if os.path.exists(alt_path):
                        img_path = alt_path
                        break
            except ValueError:
                pass
                
            # Try 6-digit format with different extensions
            for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                alt_path = os.path.join(self.img_dir, f'{formatted_id}{ext}')
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break
        
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy sample if image can't be loaded
            return {
                'image': torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32),
                'mask': torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32),
                'img_id': formatted_id,
                'valid': False
            }
        
        # Get original image size
        width, height = img.size
        
        # Load mask if required
        if self.load_mask:
            try:
                mask = self.load_mask(img_id, (height, width))
                
                # Apply transformations to both image and mask
                transformed_img = self.transform(img)
                
                # Apply mask transforms
                transformed_mask = apply_mask_transforms(mask, self.transform, img)
                
                # Prepare final mask
                final_mask = prepare_mask(transformed_mask, img_size=self.img_size)
            except Exception as e:
                print(f"Error processing mask for {img_id}: {e}")
                # If mask processing fails, use empty mask
                transformed_img = self.transform(img)
                final_mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
        else:
            # Apply transformations to image only
            transformed_img = self.transform(img)
            final_mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
        
        # Return sample
        return {
            'image': transformed_img,
            'mask': final_mask,
            'img_id': formatted_id,
            'valid': True
        }


class RSVGBboxDataset(Dataset):
    """
    Dataset for RSVG bounding box data
    
    Args:
        root_dir: Root directory of the RSVG dataset
        split: Dataset split ('train', 'val', or 'test')
        transform: Optional transforms to apply
        img_size: Target image size (default: 224)
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        img_size: int = 224,
        aug_intensity: str = 'medium'
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        # Set up image directories
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.ann_dir = os.path.join(root_dir, 'Annotations')
        
        # Get split file path
        split_file = os.path.join(root_dir, f'{split}.txt')
        
        # Load image IDs from split file
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.image_ids = [line.strip() for line in f.readlines()]
        else:
            # Fallback to torch.load if split file is not txt
            data = torch.load(os.path.join(root_dir, f'{split}.pt'), weights_only=True)
            self.image_ids = [entry[0] for entry in data]  # First element is image name
        
        # Set up transforms
        if transform is None:
            self.transform = get_transform(
                img_size=img_size,
                is_training=(split == 'train'),
                use_augmentation=True,
                aug_intensity=aug_intensity
            )
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def format_image_id(self, img_id: str) -> str:
        """Format image ID with zero padding"""
        try:
            return f"{int(img_id):06d}"
        except ValueError:
            return img_id
    
    def load_annotation(self, img_id: str) -> Tuple[str, List[Tuple[str, Tuple[int, int, int, int]]], Tuple[int, int]]:
        """Load XML annotation for an image ID"""
        formatted_id = self.format_image_id(img_id)
        xml_path = os.path.join(self.ann_dir, f'{formatted_id}.xml')
        
        if not os.path.exists(xml_path):
            print(f"XML file not found: {xml_path}")
            return None, [], (None, None)
            
        try:
            with open(xml_path, 'r') as f:
                xml_content = f.read()
            
            root = ET.fromstring(xml_content)
            filename = root.find('filename').text
            
            # Get image size
            size_elem = root.find('size')
            if size_elem is not None:
                width = int(size_elem.find('width').text)
                height = int(size_elem.find('height').text)
            else:
                width, height = None, None
            
            objects = []
            for obj in root.findall('object'):
                # Get class name
                class_name = None
                for child in obj:
                    if child.tag in ['n', 'name']:
                        if child.text and child.text.strip():
                            class_name = child.text.strip()
                            break
                
                if not class_name:
                    continue
                    
                # Get bounding box
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                    
                try:
                    xmin = int(float(bbox.find('xmin').text))
                    ymin = int(float(bbox.find('ymin').text))
                    xmax = int(float(bbox.find('xmax').text))
                    ymax = int(float(bbox.find('ymax').text))
                    
                    objects.append((class_name, (xmin, ymin, xmax, ymax)))
                except (ValueError, AttributeError):
                    continue
            
            return filename, objects, (width, height)
        except Exception as e:
            print(f"Error parsing annotation for {img_id}: {e}")
            return None, [], (None, None)
    
    def normalize_boxes(self, boxes: List[Tuple[int, int, int, int]], img_size: Tuple[int, int]) -> torch.Tensor:
        """
        Normalize bounding box coordinates to [0, 1] range
        
        Args:
            boxes: List of bounding boxes in (xmin, ymin, xmax, ymax) format
            img_size: Image size as (width, height)
        
        Returns:
            Tensor of normalized boxes in (x_center, y_center, width, height) format
        """
        width, height = img_size
        normalized_boxes = []
        
        for xmin, ymin, xmax, ymax in boxes:
            # Ensure coordinates are within image bounds
            xmin = max(0, min(xmin, width - 1))
            ymin = max(0, min(ymin, height - 1))
            xmax = max(0, min(xmax, width - 1))
            ymax = max(0, min(ymax, height - 1))
            
            # Convert to center, width, height format
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            
            normalized_boxes.append([x_center, y_center, box_width, box_height])
        
        # If no boxes, return empty tensor with shape (0, 4)
        if not normalized_boxes:
            return torch.zeros((0, 4), dtype=torch.float32)
            
        return torch.tensor(normalized_boxes, dtype=torch.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_id = self.image_ids[idx]
        formatted_id = self.format_image_id(img_id)
        
        # Load annotation
        filename, objects, (width, height) = self.load_annotation(img_id)
        
        if filename is None:
            # If annotation not found, use image ID as filename
            img_path = os.path.join(self.img_dir, f'{formatted_id}.jpg')
            
            # Also try 5-digit format (annotations use 6 digits but images might use 5)
            try:
                five_digit_id = f"{int(img_id):05d}"
                alt_img_path = os.path.join(self.img_dir, f'{five_digit_id}.jpg')
                if os.path.exists(alt_img_path):
                    img_path = alt_img_path
            except ValueError:
                pass
        else:
            img_path = os.path.join(self.img_dir, filename)
        
        # Try different file extensions if file not found
        if not os.path.exists(img_path):
            # Try 5-digit format with different extensions
            try:
                five_digit_id = f"{int(img_id):05d}"
                for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                    alt_path = os.path.join(self.img_dir, f'{five_digit_id}{ext}')
                    if os.path.exists(alt_path):
                        img_path = alt_path
                        break
            except ValueError:
                pass
                
            # Try 6-digit format with different extensions  
            for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                alt_path = os.path.join(self.img_dir, f'{formatted_id}{ext}')
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break
        
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy sample if image can't be loaded
            return {
                'image': torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32),
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'img_id': formatted_id,
                'valid': False
            }
        
        # Get image size if not available from annotation
        if width is None or height is None:
            width, height = img.size
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        
        for class_name, bbox in objects:
            boxes.append(bbox)
            # Convert class name to class ID (you might need a class mapping here)
            # For simplicity, using 0 as default class ID
            labels.append(0)
        
        # Apply transformations to image
        transformed_img = self.transform(img)
        
        # Normalize bounding boxes
        normalized_boxes = self.normalize_boxes(boxes, (width, height))
        
        # Return sample
        return {
            'image': transformed_img,
            'boxes': normalized_boxes,
            'labels': torch.tensor(labels, dtype=torch.int64),
            'img_id': formatted_id,
            'valid': True
        }


def create_rsvg_dataloader(
    root_dir: str,
    split: str = 'train',
    batch_size: int = 16,
    img_size: int = 224,
    num_workers: int = 4,
    aug_intensity: str = 'medium',
    use_masks: bool = True,
):
    """
    Create a dataloader for RSVG dataset
    
    Args:
        root_dir: Root directory of the RSVG dataset
        split: Dataset split ('train', 'val', or 'test')
        batch_size: Batch size
        img_size: Target image size
        num_workers: Number of workers for data loading
        aug_intensity: Augmentation intensity
        use_masks: Whether to use segmentation masks
    
    Returns:
        DataLoader for the RSVG dataset
    """
    from torch.utils.data import DataLoader
    
    # Create dataset
    if use_masks:
        dataset = RSVGSegmentationDataset(
            root_dir=root_dir,
            split=split,
            img_size=img_size,
            aug_intensity=aug_intensity
        )
    else:
        dataset = RSVGBboxDataset(
            root_dir=root_dir,
            split=split,
            img_size=img_size,
            aug_intensity=aug_intensity
        )
    
    # Create dataloader with custom collate function if using bounding boxes
    if not use_masks:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train'),
            collate_fn=bbox_collate_fn
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )
    
    return dataloader

def bbox_collate_fn(batch):
    """
    Custom collate function for handling variable-sized bounding boxes
    
    Args:
        batch: List of samples from the dataset
    
    Returns:
        Dict with batched samples
    """
    # Filter out invalid samples
    valid_samples = [sample for sample in batch if sample.get('valid', True)]
    
    if not valid_samples:
        # Return empty batch if no valid samples
        return {
            'image': torch.zeros((0, 3, 224, 224)),
            'boxes': torch.zeros((0, 0, 4)),
            'labels': torch.zeros((0, 0)),
            'img_id': [],
            'valid': torch.zeros((0,), dtype=torch.bool)
        }
    
    # Initialize batch dictionary
    batch_dict = {
        'image': [],
        'boxes': [],
        'labels': [],
        'img_id': [],
        'valid': []
    }
    
    # Collect data from each sample
    for sample in valid_samples:
        for key in batch_dict:
            if key in sample:
                batch_dict[key].append(sample[key])
    
    # Convert to tensors
    batch_dict['image'] = torch.stack(batch_dict['image'], dim=0)
    batch_dict['valid'] = torch.tensor(batch_dict['valid'], dtype=torch.bool)
    
    # Don't stack boxes and labels as they may have different sizes
    # They'll remain as lists of tensors
    
    return batch_dict 

class RSVGCombinedDataset(Dataset):
    """
    Dataset class for the combined RSVG dataset.
    This dataset loads data from .pth files containing lists of [image_path, _, bbox, text, _].
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size: int = 224,
        use_augmentation: bool = True,
        max_samples: Optional[int] = None,  # Limit number of samples for debugging
        prefer_5digit: bool = True          # Prefer 5-digit filenames for DIOR-RSVG
    ):
        """
        Initialize the RSVG combined dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            split: Dataset split ('train', 'val', or 'test')
            transform: Optional transform for input images
            target_transform: Optional transform for targets
            image_size: Target image size for resizing
            use_augmentation: Whether to use data augmentation
            max_samples: Maximum number of samples to load (for debugging)
            prefer_5digit: Prefer 5-digit filenames (for DIOR-RSVG compatibility)
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.max_samples = max_samples
        self.prefer_5digit = prefer_5digit
        
        # Set transforms
        if transform is None:
            self.transform = get_transform(
                img_size=image_size,
                is_training=(split == 'train'),
                use_augmentation=use_augmentation and split == 'train'
            )
        else:
            self.transform = transform
            
        self.target_transform = target_transform
        
        # Initialize data
        self.samples = []
        self.image_dir = os.path.join(self.root_dir, 'images')
        
        # Add counters for missing images
        self.missing_images_count = 0
        self.total_attempts = 0
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load data from .pth files."""
        # Path to data file
        data_file = os.path.join(self.root_dir, f'rsvg_{self.split}.pth')
        
        # Try alternative paths if the file doesn't exist
        if not os.path.exists(data_file):
            print(f"Data file not found at: {data_file}")
            alternative_paths = [
                # Try with absolute path
                os.path.join(os.path.abspath(self.root_dir), f'rsvg_{self.split}.pth'),
                # Try with just the basename of root_dir
                os.path.join(os.path.basename(self.root_dir), f'rsvg_{self.split}.pth'),
                # Try the base directory without the sat_vg prefix
                f"{self.root_dir.replace('sat_vg/', '')}/rsvg_{self.split}.pth",
                # Try the root directory with sat_vg prefix
                f"sat_vg/{self.root_dir}/rsvg_{self.split}.pth",
                # Try finding the file in the current directory
                f"rsvg_{self.split}.pth",
                # Try finding relative to the project root
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                             self.root_dir, f'rsvg_{self.split}.pth'),
                # For home directory
                os.path.expanduser(f"~/Trans-VG/{self.root_dir}/rsvg_{self.split}.pth"),
                os.path.expanduser(f"~/Trans-VG/sat_vg/{self.root_dir}/rsvg_{self.split}.pth"),
            ]
            
            for alt_path in alternative_paths:
                print(f"Trying alternative path: {alt_path}")
                if os.path.exists(alt_path):
                    data_file = alt_path
                    print(f"Found data file at: {data_file}")
                    break
        
        if not os.path.exists(data_file):
            print(f"Error: Could not find data file at {data_file} or any alternative locations.")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Specified root_dir: {self.root_dir}")
            
            possible_files = []
            for root, dirs, files in os.walk(os.getcwd(), topdown=True):
                for file in files:
                    if file == f'rsvg_{self.split}.pth':
                        possible_files.append(os.path.join(root, file))
            
            if possible_files:
                print(f"Found potential data files:")
                for file in possible_files:
                    print(f"  - {file}")
                
                # Use the first one found
                data_file = possible_files[0]
                print(f"Using: {data_file}")
            else:
                raise FileNotFoundError(f"Data file {data_file} not found")
        
        # Loading data with better error handling
        try:
            # First try with weights_only parameter (for newer PyTorch versions)
            try:
                data = torch.load(data_file, weights_only=True, map_location='cpu')
            except TypeError:
                # Fall back to standard loading for older PyTorch versions
                data = torch.load(data_file, map_location='cpu')
                
            print(f"Loaded {len(data)} samples from {data_file}")
            
            # Validate the first sample to make sure the format is correct
            if len(data) > 0:
                first_sample = data[0]
                if not (isinstance(first_sample, list) and len(first_sample) >= 5):
                    print(f"Warning: Data format doesn't match expected [img_path, _, bbox, text, _] format")
                else:
                    print(f"Sample format: {[type(item) for item in first_sample]}")
            
            # Limit number of samples if requested
            if self.max_samples is not None and self.max_samples < len(data):
                data = data[:self.max_samples]
                print(f"Limited to {self.max_samples} samples for debugging")
            
            # Filter out samples with invalid format
            valid_samples = []
            for sample in data:
                if (isinstance(sample, list) and len(sample) >= 5 and 
                    isinstance(sample[0], str) and
                    isinstance(sample[2], list) and len(sample[2]) == 4):
                    valid_samples.append(sample)
                else:
                    print(f"Skipping invalid sample: {sample}")
            
            if len(valid_samples) < len(data):
                print(f"Filtered out {len(data) - len(valid_samples)} invalid samples")
            
            # Store data
            self.samples = valid_samples
            
        except Exception as e:
            print(f"Error loading data from {data_file}: {str(e)}")
            # Initialize with empty list
            self.samples = []
            # Re-raise the exception with more information
            raise Exception(f"Failed to load dataset from {data_file}: {str(e)}")
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image, target_dict)
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} is out of bounds for dataset with {len(self.samples)} samples")
        
        # Get sample data
        sample = self.samples[idx]
        
        # Parse sample data
        img_path, _, bbox, text, _ = sample
        
        # Check if the image path is valid
        # If it's relative, try different paths to find the actual image
        if not os.path.isabs(img_path):
            # Potential image paths to try
            potential_paths = [
                # Original path
                os.path.join(self.image_dir, img_path),
                # Try with just the basename
                os.path.join(self.image_dir, os.path.basename(img_path)),
                # Same directory as data file
                os.path.join(os.path.dirname(self.root_dir), "images", img_path),
                # Project root directory
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            "images", img_path),
                # Path relative to the project root
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            self.root_dir, "images", img_path),
                # With sat_vg prefix
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            "sat_vg", self.root_dir, "images", img_path),
                # Try the standalone path
                img_path
            ]
            
            # Try 5-digit variations if enabled
            basename, ext = os.path.splitext(os.path.basename(img_path))
            if self.prefer_5digit and basename.isdigit():
                five_digit = f"{int(basename):05d}{ext}"
                potential_paths.extend([
                    os.path.join(self.image_dir, five_digit),
                    os.path.join(os.path.dirname(self.root_dir), "images", five_digit),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                "images", five_digit),
                    five_digit,
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                self.root_dir, "images", five_digit),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                "sat_vg", self.root_dir, "images", five_digit)
                ])
            
            # Try different extensions if needed
            for try_ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                # Add versions with different extensions
                potential_paths.extend([
                    os.path.join(self.image_dir, f"{basename}{try_ext}"),
                    os.path.join(os.path.dirname(self.root_dir), "images", f"{basename}{try_ext}")
                ])
                
                # Try 5-digit format with different extensions
                if basename.isdigit():
                    potential_paths.extend([
                        os.path.join(self.image_dir, f"{int(basename):05d}{try_ext}"),
                        os.path.join(os.path.dirname(self.root_dir), "images", f"{int(basename):05d}{try_ext}")
                    ])
            
            # Try each path until we find a valid image
            full_path = None
            for path in potential_paths:
                if path and os.path.exists(path):
                    full_path = path
                    break
            
            if full_path is None:
                # If no valid path found after trying all alternatives, use original as fallback
                # (will likely fail, but keeps the original behavior)
                full_path = os.path.join(self.image_dir, img_path)
        else:
            full_path = img_path
        
        # Increment total attempts counter
        self.total_attempts += 1
        
        # Load image
        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            # Don't print individual errors, just count them
            self.missing_images_count += 1
            
            # Print summary periodically
            if self.missing_images_count == 1 or self.missing_images_count % 100 == 0:
                print(f"Missing images: {self.missing_images_count}/{self.total_attempts} "
                      f"({self.missing_images_count/self.total_attempts:.1%})")
            
            # Return a blank image as fallback
            image = Image.new('RGB', (self.image_size, self.image_size), color=0)
        
        # Apply transforms to the image
        if self.transform:
            image = self.transform(image)
        
        # Create target dictionary
        target = {
            'bbox': torch.tensor(bbox, dtype=torch.float32),
            'text': text
        }
        
        # Apply target transform if available
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target
    
    def get_stats(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'num_samples': len(self.samples),
            'split': self.split
        }

    def collate_fn(self, batch):
        """
        Custom collate function that handles the dictionary targets correctly.
        
        Args:
            batch: List of (image, target) tuples
            
        Returns:
            Tuple of (images, targets)
        """
        images = torch.stack([item[0] for item in batch])
        
        # For targets, we want to keep them as a list of dictionaries
        targets = []
        for item in batch:
            target = item[1]
            # Ensure target is a dictionary
            if not isinstance(target, dict):
                target = {'bbox': torch.zeros(4), 'text': str(target)}
            targets.append(target)
        
        return images, targets


def create_rsvg_combined_dataloader(
    root_dir: str,
    split: str = 'train',
    batch_size: int = 16,
    img_size: int = 224,
    num_workers: int = 4,
    use_augmentation: bool = True,
    max_samples: Optional[int] = None,
    prefer_5digit: bool = True
):
    """Create a dataloader for the RSVG combined dataset"""
    # Create dataset
    dataset = RSVGCombinedDataset(
        root_dir=root_dir,
        split=split,
        image_size=img_size,
        use_augmentation=use_augmentation,
        max_samples=max_samples,
        prefer_5digit=prefer_5digit
    )
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        collate_fn=dataset.collate_fn
    )
    
    return dataloader 