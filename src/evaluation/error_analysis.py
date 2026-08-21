import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import sys
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import load_config
from src.data.preprocess import prepare_datasets
from src.data.dataloader import get_dataloaders
from src.models.factory import create_model
from src.inference.explain import generate_heatmap
from src.data.augmentations import get_albumentations_transforms

def run_error_analysis(config, model_path=None, top_n=20):
    """
    Анализирует ошибки модели на тестовой выборке (Ложно-положительные и Ложно-отрицательные).
    Находит "уверенные ошибки" (High Confidence errors), визуализирует их через Grad-CAM
    и сохраняет подробный отчет.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Готовим данные (нам нужен только test)
    _, _, test_ds = prepare_datasets(config)
    _, _, test_loader = get_dataloaders(config, test_ds, test_ds, test_ds)
    
    # 2. Инициализируем модель
    model_path = model_path or config.get('inference', {}).get('model_path', 'models/best.pth')
    
    cfg_copy = config.copy()
    cfg_copy['model'] = config['model'].copy()
    cfg_copy['model']['pretrained'] = False
    
    print(f"Загрузка модели из {model_path}...")
    model = create_model(cfg_copy)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Инференс на всей тестовой выборке
    all_preds = []
    all_probs = []
    all_targets = []
    
    print("Инференс на тестовой выборке...")
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Оценка"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Сохраняем вероятность именно ПРЕДСКАЗАННОГО класса (чтобы понять уверенность ошибки)
            confidences = probs[torch.arange(len(preds)), preds]
            
            all_targets.extend(targets.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(confidences.cpu().numpy())
            
    # 4. Формируем DataFrame
    results_df = pd.DataFrame({
        'path': test_ds.df['path'].values,
        'actual': all_targets,
        'predicted': all_preds,
        'confidence': all_probs
    })
    
    # Фильтруем только ошибочные предсказания
    errors_df = results_df[results_df['actual'] != results_df['predicted']].copy()
    
    def get_error_type(row):
        # Если бинарная классификация: 0=Норма, 1=ROP
        if row['actual'] == 0 and row['predicted'] == 1:
            return 'False Positive'
        elif row['actual'] == 1 and row['predicted'] == 0:
            return 'False Negative'
        return 'Misclassified'
        
    errors_df['error_type'] = errors_df.apply(get_error_type, axis=1)
    
    # Сортируем по уверенности (от максимальной к минимальной)
    errors_df = errors_df.sort_values(by='confidence', ascending=False)
    
    out_dir = project_root / 'reports' / 'hard_examples'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем табличный отчет
    report_path = out_dir / 'error_report.csv'
    errors_df.to_csv(report_path, index=False)
    print(f"\nНайдено {len(errors_df)} ошибок. Отчет сохранен в {report_path}")
    
    # 5. Визуализация "Самых уверенных ошибок"
    top_errors = errors_df.head(top_n)
    
    arch_name = config['model'].get('architecture', '')
    img_size = config['training'].get('image_size', 384)
    _, val_tf = get_albumentations_transforms(img_size)
    
    print(f"\nГенерация тепловых карт (Grad-CAM) для Топ-{len(top_errors)} сложных примеров...")
    
    saved_count = 0
    for idx, row in tqdm(top_errors.iterrows(), total=len(top_errors)):
        img_path = str(row['path'])
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue
            
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # Подготовка тензора
        transformed = val_tf(image=orig_img)['image']
        if isinstance(transformed, torch.Tensor):
            input_tensor = transformed.unsqueeze(0).to(device)
        else:
            input_tensor = torch.tensor(transformed).unsqueeze(0).to(device)
            
        # Генерация CAM
        cam_img = generate_heatmap(
            model=model,
            input_tensor=input_tensor,
            orig_image=orig_img,
            arch_name=arch_name,
            method='GradCAM' # Можно поменять на GradCAM++
        )
        
        if cam_img is not None:
            # Формируем информативное имя файла
            safe_conf = f"{row['confidence']:.3f}".replace('.', 'p')
            safe_name = Path(img_path).stem
            filename = f"{row['error_type'].replace(' ', '_')}_conf{safe_conf}_{safe_name}.png"
            
            # Конвертируем обратно в BGR для cv2.imwrite
            save_img = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / filename), save_img)
            saved_count += 1
            
    print(f"Успешно сохранено {saved_count} изображений в {out_dir}")
    return errors_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ROP Error Analysis")
    parser.add_argument('--config', default='configs/base.yaml', help='Path to config file')
    parser.add_argument('--model', default=None, help='Path to weights (.pth)')
    parser.add_argument('--top', type=int, default=20, help='Number of hard examples to visualize')
    
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_error_analysis(cfg, model_path=args.model, top_n=args.top)
