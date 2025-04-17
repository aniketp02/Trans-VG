import os
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer
from torchvision import transforms


class SatVGDataset(data.Dataset):
    """
    Satellite Visual Grounding Dataset
    
    This dataset loads images and text queries for satellite visual grounding,
    processes them, and returns them in the format expected by the SatVG model.
    """
    def __init__(self, data_root, split='train', max_query_len=40, 
                 bert_model='bert-base-uncased', img_size=640,
                 use_augmentation=True):
        """
        Initialize the dataset.
        
        Args:
            data_root: Path to the dataset
            split: Dataset split ('train', 'val', or 'test')
            max_query_len: Maximum query length for BERT tokenization
            bert_model: BERT model name
            img_size: Image size
            use_augmentation: Whether to use data augmentation
        """
        self.data_root = data_root
        self.split = split
        self.max_query_len = max_query_len
        self.img_size = img_size
        
        # Load dataset
        self.data_path = os.path.join(data_root, f'rsvg_{split}.pth')
        self.data = torch.load(self.data_path, weights_only=True)
        self.im_dir = os.path.join(data_root, 'images')
        
        print(f"Loaded {len(self.data)} samples for {split} split")
        
        # Set up augmentation based on split
        self.use_augmentation = use_augmentation and split == 'train'
        
        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        
        # Set up image transforms
        if self.use_augmentation:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        """Get the number of samples in the dataset."""
        return len(self.data)
    
    def tokenize_text(self, text):
        """
        Tokenize text using BERT tokenizer.
        
        Args:
            text: Input text string
            
        Returns:
            input_ids: Token IDs
            attention_mask: Attention mask
        """
        # Tokenize
        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_query_len,
            return_tensors='pt'
        )
        
        return tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0)
    
    def load_image(self, image_path):
        """
        Load and preprocess image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Preprocessed image tensor
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            return torch.zeros(3, self.img_size, self.img_size)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - img: Image tensor
                - text_tokens: Tokenized text
                - text_mask: Attention mask for text
                - target: Target bounding box in [x_center, y_center, width, height] format
                - original_bbox: Original bounding box for evaluation
                - image_id: Image identifier
        """
        sample = self.data[idx]
        
        # Extract data from the sample
        if isinstance(sample, list) and len(sample) >= 4:
            # Format: [img_file, None, bbox, phrase, None]
            image_name = sample[0]
            bbox = sample[2]
            text = sample[3]
        elif isinstance(sample, tuple) and len(sample) >= 3:
            # Format: (image_name, bbox, expression)
            image_name = sample[0]
            bbox = sample[1]
            text = sample[2]
        else:
            raise ValueError(f"Unexpected sample format: {type(sample)}")
        
        # Load image
        image_path = os.path.join(self.im_dir, image_name)
        img = self.load_image(image_path)
        
        # Tokenize text
        text_tokens, text_mask = self.tokenize_text(text)
        
        # Process bounding box to [x_center, y_center, width, height] format in [0, 1] range
        if len(bbox) == 4:
            # Convert from [x1, y1, x2, y2] to [x_center, y_center, width, height]
            x1, y1, x2, y2 = bbox
            
            # Make sure the bounding box is valid
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Convert to normalized coordinates
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Normalize coordinates to [0, 1]
            x_center /= self.img_size
            y_center /= self.img_size
            width /= self.img_size
            height /= self.img_size
            
            # Create target tensor
            target = torch.tensor([x_center, y_center, width, height], dtype=torch.float)
        else:
            raise ValueError(f"Unexpected bbox format: {bbox}")
        
        # Return a dictionary with all the necessary data
        return {
            'img': img,
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'target': target,
            'original_bbox': torch.tensor(bbox, dtype=torch.float),
            'image_id': image_name
        }


def collate_fn(batch):
    """
    Collate function for data loader.
    
    Args:
        batch: List of samples
        
    Returns:
        Dictionary containing batched data
    """
    images = torch.stack([item['img'] for item in batch])
    text_tokens = torch.stack([item['text_tokens'] for item in batch])
    text_masks = torch.stack([item['text_mask'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    original_bboxes = torch.stack([item['original_bbox'] for item in batch])
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'img': images,
        'text_tokens': text_tokens,
        'text_mask': text_masks,
        'target': targets,
        'original_bbox': original_bboxes,
        'image_id': image_ids
    } 