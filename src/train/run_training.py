import argparse
import sys
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Добавляем корень проекта в sys.path для корректных импортов
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import load_config
from src.data.preprocess import prepare_datasets
from src.data.dataloader import get_dataloaders
from src.models.factory import create_model
from src.train.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="OptiX Training Pipeline")
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Путь к конфигурационному файлу')
    
    # Аргументы для переопределения конфига
    parser.add_argument('--arch', type=str, help='Архитектура модели (например, efficientnet-b2)')
    parser.add_argument('--epochs', type=int, help='Количество эпох обучения')
    parser.add_argument('--batch-size', type=int, help='Размер батча (batch size)')
    parser.add_argument('--lr', type=float, help='Learning rate (шаг обучения)')
    parser.add_argument('--freeze-backbone', action='store_true', help='Заморозить бэкбоун модели (обучать только классификатор)')
    parser.add_argument('--loss', type=str, choices=['ce', 'focal', 'weighted_ce'], help='Тип функции потерь')
    
    return parser.parse_args()

def update_config_from_args(config, args):
    """Перезаписывает значения конфига, если они переданы через командную строку."""
    if args.arch:
        config['model']['architecture'] = args.arch
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.freeze_backbone:
        config['model']['freeze_backbone'] = True
    if args.loss:
        config['training']['loss_type'] = args.loss
    return config

def main():
    args = parse_args()
    
    print("=== Загрузка конфигурации ===")
    config = load_config(args.config)
    config = update_config_from_args(config, args)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Устройство для обучения: {device}")
    
    print("\n=== Подготовка данных ===")
    train_ds, val_ds, test_ds = prepare_datasets(config)
    print(f"Размер выборок -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    print("\n=== Инициализация DataLoaders ===")
    train_loader, val_loader, test_loader = get_dataloaders(config, train_ds, val_ds, test_ds)
    
    print(f"\n=== Создание модели ({config['model']['architecture']}) ===")
    model = create_model(config)
    
    print("\n=== Настройка оптимизатора и планировщика (AdamW + CosineAnnealing) ===")
    lr = config['training'].get('learning_rate', 0.001)
    
    # Передаем в оптимизатор только те параметры, которые не заморожены
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    
    epochs = config['training'].get('epochs', 50)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("\n=== Запуск тренировочного цикла ===")
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    result = trainer.fit()
    
    print("\n=== Обучение завершено! ===")
    print(f"Лучшие веса сохранены по пути: {result['model_path']}")
    print("Лучшие валидационные метрики:")
    for k, v in result['best_metrics'].items():
        if isinstance(v, float):
            print(f"  - {k}: {v:.4f}")
        else:
            print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()
