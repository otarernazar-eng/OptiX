import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

class RetinopathyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
        # Create label mapping
        self.classes = sorted(df['label'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row.get('image_path', row.get('path'))
        project_root = Path(__file__).resolve().parents[2]
        if not Path(img_path).is_absolute():
            img_path = str(project_root / img_path)

        label = self.class_to_idx[row['label']]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

def generate_eda_reports(train_df, val_df, test_df, output_dir):
    """Generates class distribution plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    splits = ['Train', 'Validation', 'Test']
    dfs = [train_df, val_df, test_df]
    
    dist_data = []
    for split, df in zip(splits, dfs):
        counts = df['label'].value_counts().reset_index()
        counts.columns = ['Label', 'Count']
        counts['Split'] = split
        dist_data.append(counts)
        
    dist_df = pd.concat(dist_data)
    
    sns.barplot(data=dist_df, x='Split', y='Count', hue='Label')
    plt.title('Распределение классов по выборкам (Train/Val/Test)')
    plt.ylabel('Количество снимков')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'class_distribution.png')
    plt.close()

def prepare_datasets(config):
    """
    Main function called from config/train script to prepare datasets.
    Reads metadata, splits it, saves splits, generates EDA, and returns PyTorch Datasets.
    """
    project_root = Path(__file__).resolve().parents[2]
    
    raw_dir = project_root / config['data'].get('raw_dir', 'data/raw')
    splits_dir = project_root / config['data'].get('splits_dir', 'data/splits')
    reports_dir = project_root / 'reports' / 'eda'
    
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = splits_dir / 'train.csv'
    val_path = splits_dir / 'val.csv'
    test_path = splits_dir / 'test.csv'
    
    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError(f"Split CSVs not found in {splits_dir}. Please run scripts/merge_datasets.py first.")
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Generate EDA
    generate_eda_reports(train_df, val_df, test_df, reports_dir)
    
    # Setup Transforms & Datasets
    from src.data.augmentations import get_albumentations_transforms, visualize_augmentations
    
    img_size = config['training'].get('image_size', 384)
    train_tf, val_tf = get_albumentations_transforms(img_size)
    
    train_dataset = RetinopathyDataset(train_df, transform=train_tf)
    val_dataset = RetinopathyDataset(val_df, transform=val_tf)
    test_dataset = RetinopathyDataset(test_df, transform=val_tf)
    
    # Save visual examples of augmentations
    vis_path = project_root / 'reports' / 'aug_examples.png'
    visualize_augmentations(train_dataset, output_path=vis_path, num_samples=5)
    
    return train_dataset, val_dataset, test_dataset
