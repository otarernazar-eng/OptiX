import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.models.factory import create_model

def test_model_inference_dimensions():
    config = {
        'model': {
            'architecture': 'efficientnet-b0',
            'num_classes': 2,
            'pretrained': False, # отключено для скорости тестов
            'freeze_backbone': True
        }
    }
    
    # Инициализация модели через фабрику
    model = create_model(config)
    model.eval()
    
    batch_size = 4
    image_size = 384
    
    # Создаем dummy тензор, имитирующий реальный выход DataLoader'а
    dummy_input = torch.randn(batch_size, 3, image_size, image_size)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    # Проверка: выход модели должен иметь размерность [batch_size, num_classes]
    assert output.shape == (batch_size, 2), f"Ожидалась форма (4, 2), получено {output.shape}"
