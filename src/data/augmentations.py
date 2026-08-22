import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch

def get_albumentations_transforms(image_size):
    """
    Returns albumentations transforms for training and validation.
    """
    train_transform = A.Compose([
        # Поддержание пропорций (padding)
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, fill=0),
        
        # Пространственные трансформации
        A.Rotate(limit=30, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=0, p=0.5),
        
        # Цветовые и шумовые трансформации
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(p=0.3),
        
        # Нормализация ImageNet и конвертация в Tensor
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        # Только resize с сохранением пропорций и нормализация
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, fill=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

def visualize_augmentations(dataset, output_path="reports/aug_examples.png", num_samples=5):
    """
    Visualizes original and augmented examples side by side and saves to disk.
    Assumes dataset applies albumentations and returns (tensor, label).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Статистика ImageNet для денормализации
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Количество сэмплов не должно превышать размер датасета
    num_samples = min(num_samples, len(dataset))
    if num_samples == 0:
        return
        
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
        
    for i in range(num_samples):
        # Достаем путь к оригинальному изображению из метаданных
        row = dataset.df.iloc[i]
        img_path = row.get('image_path', row.get('path'))
        
        # Absolute path resolution
        project_root = Path(__file__).resolve().parents[2]
        if not Path(img_path).is_absolute():
            img_path = str(project_root / img_path)
        
        # Загружаем оригинал (до аугментаций)
        orig_img = cv2.imread(img_path)
        if orig_img is not None:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        else:
            orig_img = np.zeros((100, 100, 3), dtype=np.uint8) # Fallback
        
        # Получаем аугментированный тензор из датасета
        aug_tensor, label_idx = dataset[i]
        
        # Денормализация для визуализации
        aug_img = aug_tensor.numpy().transpose(1, 2, 0)
        aug_img = std * aug_img + mean
        aug_img = np.clip(aug_img, 0, 1)
        
        axes[i][0].imshow(orig_img)
        axes[i][0].set_title(f"Original (Label: {row['label']})")
        axes[i][0].axis('off')
        
        axes[i][1].imshow(aug_img)
        axes[i][1].set_title("Augmented")
        axes[i][1].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Augmentation examples saved to {output_path}")
