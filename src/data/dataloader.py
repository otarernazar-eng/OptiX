import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

def create_weighted_sampler(dataset):
    """
    Creates a WeightedRandomSampler for handling class imbalance.
    Works transparently for both binary and multi-class classification tasks.
    """
    # Extract labels from the dataset's dataframe
    df = dataset.df
    labels = df['label'].values
    
    # Calculate class counts
    class_counts = df['label'].value_counts().to_dict()
    num_samples = len(labels)
    
    # Weight for each class = total_samples / class_count
    # This gives higher weight to rare classes
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
    
    # Assign a weight to each individual sample in the dataset based on its class
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=num_samples,
        replacement=True
    )
    
    return sampler

def get_dataloaders(config, train_dataset, val_dataset, test_dataset):
    """
    Creates and returns PyTorch DataLoaders for train, val, and test splits.
    Reads batch_size, num_workers, and sampling preferences from config.
    """
    batch_size = config['training'].get('batch_size', 16)
    num_workers = config['training'].get('num_workers', 4)
    use_sampler = config['training'].get('use_weighted_sampler', False)
    
    # Prepare sampler for training if requested
    train_sampler = None
    shuffle = True
    
    if use_sampler:
        train_sampler = create_weighted_sampler(train_dataset)
        shuffle = False # shuffle must be False if a sampler is specified
        print("WeightedRandomSampler enabled for training dataloader (Class Balancing ON).")
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True, # Speeds up transfer to GPU
        drop_last=True   # Drops the last incomplete batch to stabilize BatchNorm
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader
