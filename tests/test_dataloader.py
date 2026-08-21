import pytest
import pandas as pd
import numpy as np
import cv2
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.augmentations import get_albumentations_transforms
from src.data.preprocess import RetinopathyDataset
from src.data.dataloader import get_dataloaders

@pytest.fixture
def dummy_dataset(tmp_path):
    img_paths = []
    # Генерируем 10 фейковых изображений
    for i in range(10):
        path = tmp_path / f"img{i}.png"
        cv2.imwrite(str(path), np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img_paths.append(str(path))
        
    # Искусственный дисбаланс классов (7 к 3), чтобы неявно проверить инициализацию Sampler'а
    df = pd.DataFrame({
        'path': img_paths,
        'label': ['Normal'] * 7 + ['ROP'] * 3
    })
    
    _, val_tf = get_albumentations_transforms(384)
    return RetinopathyDataset(df, transform=val_tf)

def test_dataloader_batch_shapes(dummy_dataset):
    config = {
        'training': {
            'batch_size': 4,
            'num_workers': 0,
            'use_weighted_sampler': True
        }
    }
    
    train_loader, _, _ = get_dataloaders(config, dummy_dataset, dummy_dataset, dummy_dataset)
    
    # Получаем первый батч
    inputs, targets = next(iter(train_loader))
    
    # 1. Проверка размерности батча изображений: [Batch Size, Channels, Height, Width]
    assert inputs.shape == (4, 3, 384, 384), f"Неверная размерность батча {inputs.shape}"
    
    # 2. Проверка размерности таргетов: [Batch Size]
    assert targets.shape == (4,), f"Неверная размерность таргетов {targets.shape}"
