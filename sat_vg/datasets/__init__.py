from .data_loader import SatVGDataset, collate_fn

def build_dataset(args, split):
    """
    Build a dataset for the specified split.
    
    Args:
        args: Arguments for dataset configuration
        split: Dataset split ('train', 'val', or 'test')
        
    Returns:
        SatVGDataset instance
    """
    return SatVGDataset(
        data_root=args.data_root,
        split=split,
        max_query_len=args.max_query_len,
        bert_model=args.bert_model,
        img_size=args.img_size,
        use_augmentation=args.use_augmentation if split == 'train' else False
    ) 