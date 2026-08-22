import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import seaborn as sns

project_root = Path(__file__).resolve().parents[1]
raw_dir = project_root / 'data' / 'raw'
splits_dir = project_root / 'data' / 'splits'
reports_dir = project_root / 'reports'

def get_label(dataset, filepath):
    path_str = str(filepath).lower()
    if dataset == 'ridirp':
        return 0 if 'normal' in path_str else 1
    elif dataset == 'farfum':
        return 0 if 'normal' in path_str else 1
    elif dataset == 'macretina':
        # Ridge is ROP pathology, others normal
        if 'ridge' in path_str:
            return 1
        return 0
    return 0

def mock_dataset_if_empty():
    """Синтезируем метаданные, если данные еще не скачаны вручную."""
    data = []
    # RIDIRP: 6004 (assume 4000 normal, 2004 ROP)
    for i in range(6004):
        data.append({'image_path': f'data/raw/ridirp/img_{i}.jpg', 'label': 0 if i < 4000 else 1, 'dataset_source': 'RIDIRP'})
    # Macretina: 1432 (Ridge 465 -> 1, OD/BV -> 0)
    for i in range(1432):
        data.append({'image_path': f'data/raw/macretina/img_{i}.jpg', 'label': 1 if i < 465 else 0, 'dataset_source': 'Macretina'})
    # FARFUM: 1533 (assume 1000 normal, 533 plus)
    for i in range(1533):
        data.append({'image_path': f'data/raw/farfum/img_{i}.jpg', 'label': 0 if i < 1000 else 1, 'dataset_source': 'FARFUM'})
    return pd.DataFrame(data)

def main():
    splits_dir.mkdir(exist_ok=True, parents=True)
    reports_dir.mkdir(exist_ok=True, parents=True)
    
    data = []
    datasets = ['ridirp', 'macretina', 'farfum']
    
    # 1. Сканирование папок
    for ds in datasets:
        ds_path = raw_dir / ds
        if ds_path.exists():
            files = list(ds_path.rglob('*.*'))
            for filepath in files:
                if filepath.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    label = get_label(ds, filepath)
                    data.append({
                        'image_path': str(filepath.relative_to(project_root)).replace('\\', '/'),
                        'label': label,
                        'dataset_source': ds.upper()
                    })
    
    if len(data) > 0:
        df = pd.DataFrame(data)
        print(f"Найдено реальных изображений: {len(df)}")
    else:
        print("Реальные данные не найдены. Используется генерация Mock-метаданных для визуализации...")
        df = mock_dataset_if_empty()
        
    # 2. Визуализация распределения
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='dataset_source', hue='label', palette='Set2')
    plt.title('Распределение классов по датасетам (0 = Норма, 1 = ROP Патология)')
    plt.xlabel('Датасет')
    plt.ylabel('Количество снимков')
    
    plot_path = reports_dir / 'dataset_distribution.png'
    plt.savefig(plot_path)
    print(f"График распределения сохранен в {plot_path}")
    
    # 3. Разделение на train/val/test со стратификацией
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)
    
    train_df.to_csv(splits_dir / 'train.csv', index=False)
    val_df.to_csv(splits_dir / 'val.csv', index=False)
    test_df.to_csv(splits_dir / 'test.csv', index=False)
    print("CSV-файлы разбиения сохранены в data/splits/ (train.csv, val.csv, test.csv)")
    
    # 4. Анализ дисбаланса
    total = len(df)
    positives = df['label'].sum()
    negatives = total - positives
    print(f"\nАнализ объединенного датасета:")
    print(f"Всего снимков: {total}")
    print(f"Норма (Класс 0): {negatives} | ROP (Класс 1): {positives}")
    
    if abs(positives - negatives) / total > 0.2:
        print("⚠️ Внимание: Сильный дисбаланс классов.")
        print("В Trainer автоматически будет использован WeightedRandomSampler.")

if __name__ == "__main__":
    main()
