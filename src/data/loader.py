import os
import urllib.request
import pandas as pd
from pathlib import Path
from PIL import Image

DATASETS = {
    'i-ROP': 'https://example.com/i-rop.zip',
    'RIDIRP': 'https://example.com/ridirp.zip',
    'Macretina': 'https://example.com/macretina.zip'
}

def check_integrity(file_path, expected_formats=('.png', '.jpg', '.jpeg')):
    """Validates the image file."""
    try:
        if not str(file_path).lower().endswith(expected_formats):
            return False
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def prepare_data(config, progress_callback=None):
    """Downloads requested datasets and collects metadata."""
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / config['data'].get('raw_dir', 'data/raw')
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (dataset_name, url) in enumerate(DATASETS.items()):
        if progress_callback:
            progress_callback(f"Попытка загрузки {dataset_name}...", i / len(DATASETS))
            
        dataset_path = raw_dir / dataset_name
        dataset_path.mkdir(exist_ok=True)
        
        try:
            # ЗДЕСЬ должна быть логика скачивания zip-архива
            pass
        except Exception as e:
            print(f"Ошибка загрузки {dataset_name}: {e}")
            
    if progress_callback:
        progress_callback("Сбор метаданных и проверка целостности (Integrity check)...", 0.9)
        
    valid_metadata = []
    # Сканируем все скачанные файлы
    for filepath in raw_dir.rglob('*'):
        if filepath.is_file() and filepath.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            if check_integrity(filepath):
                # Простая логика определения меток (можно усложнить в зависимости от структуры датасета)
                label = 'ROP' if 'rop' in filepath.name.lower() else 'Normal'
                valid_metadata.append({
                    'filename': filepath.name,
                    'dataset': filepath.parent.name,
                    'label': label,
                    'path': str(filepath),
                    'size_bytes': filepath.stat().st_size
                })
            else:
                try:
                    os.remove(filepath)
                except:
                    pass
                
    df = pd.DataFrame(valid_metadata)
    csv_path = raw_dir / "metadata.csv"
    # Сохраняем, даже если пусто (пустой датафрейм)
    df.to_csv(csv_path, index=False)
    
    if progress_callback:
        progress_callback("Готово!", 1.0)
        
    return str(csv_path)

if __name__ == "__main__":
    from src.utils.config import load_config
    cfg = load_config()
    prepare_data(cfg)
