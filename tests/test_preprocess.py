import pytest
import pandas as pd
import numpy as np
import cv2
import sys
from pathlib import Path

# Добавляем корень проекта для правильных импортов
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.augmentations import get_albumentations_transforms
from src.data.preprocess import RetinopathyDataset

@pytest.fixture
def dummy_data(tmp_path):
    img1_path = tmp_path / "img1.png"
    img2_path = tmp_path / "img2.png"
    
    # Генерируем тестовые прямоугольные изображения, чтобы проверить Padding логику
    cv2.imwrite(str(img1_path), np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8))
    cv2.imwrite(str(img2_path), np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8))
    
    df = pd.DataFrame({
        'path': [str(img1_path), str(img2_path)],
        'label': ['Normal', 'ROP']
    })
    return df

def test_preprocessing_sizes_and_normalization(dummy_data):
    image_size = 384
    _, val_tf = get_albumentations_transforms(image_size)
    
    # Используем валидационный трансформ, чтобы результаты были детерминированы (без рандомных аугментаций)
    dataset = RetinopathyDataset(dummy_data, transform=val_tf)
    
    img_tensor, label = dataset[0]
    
    # 1. Проверка правильного размера после ресайза и паддинга (должен стать квадратом 384х384)
    assert img_tensor.shape == (3, image_size, image_size), f"Ожидался размер (3, {image_size}, {image_size}), получено {img_tensor.shape}"
    
    # 2. Проверка нормализации (ImageNet normalization)
    # Исходные значения пикселей 0-255 должны быть нормализованы (обычно диапазон от -2.5 до 2.5)
    assert img_tensor.min() >= -3.0
    assert img_tensor.max() <= 3.0
    
    # 3. Проверка типа метки
    assert isinstance(label, int)
