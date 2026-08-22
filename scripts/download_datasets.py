import os
from pathlib import Path

def print_manual_instructions():
    print("=" * 60)
    print("ОШИБКА АВТОМАТИЗАЦИИ: Прямые ссылки на скачивание защищены Figshare API.")
    print("Пожалуйста, загрузите датасеты вручную по следующим инструкциям:\n")
    
    print("1. RIDIRP (Чехия) -> data/raw/ridirp/")
    print("   Ссылка: https://doi.org/10.6084/m9.figshare.c.6626162.v1")
    print("   Описание: 6004 изображения. Скачайте zip-архивы и извлеките их в папку ridirp.\n")
    
    print("2. Macretina (Индия) -> data/raw/macretina/")
    print("   Ссылка: Поиск на https://figshare.com по запросу 'Macretina'")
    print("   Описание: 1432 изображения. Вам нужны папки Ridge (465), OD (500), BV (38).\n")
    
    print("3. FARFUM-RoP (Иран) -> data/raw/farfum/")
    print("   Ссылка: https://doi.org/10.6084/m9.figshare.c.6721269.v2")
    print("   Описание: 1533 изображения с аннотациями Normal, Pre-Plus, Plus.\n")
    print("=" * 60)
    print("После загрузки данных запустите скрипт объединения: python scripts/merge_datasets.py")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / 'data' / 'raw'
    
    # Создаем структуру директорий
    for ds in ['ridirp', 'macretina', 'farfum']:
        (raw_dir / ds).mkdir(parents=True, exist_ok=True)
        print(f"Директория создана: {raw_dir / ds}")
        
    print_manual_instructions()
