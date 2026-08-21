import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, auc
import sys

# Подключаем корень проекта
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.factory import create_model
from src.data.preprocess import prepare_datasets
from src.data.dataloader import get_dataloaders
from torch.cuda.amp import autocast

def evaluate_model(model, data_loader, device='cuda', use_amp=True):
    """Выполняет инференс модели на даталоадере и возвращает предсказания."""
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)

def compare_models(config, models_info):
    """
    Сравнивает список моделей на тестовой выборке.
    models_info: список словарей вида [{'name': 'Model1', 'path': '...', 'arch': 'efficientnet-b0'}, ...]
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Готовим только тестовый датасет
    _, _, test_ds = prepare_datasets(config)
    # Передаем test_ds три раза, но забираем только 3-й лоадер (test_loader)
    _, _, test_loader = get_dataloaders(config, test_ds, test_ds, test_ds)
    
    reports_dir = project_root / 'reports' / 'comparison'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    roc_data = {}
    pr_data = {}
    
    sns.set_theme(style="whitegrid")
    
    for m_info in models_info:
        name = m_info['name']
        path = m_info['path']
        arch = m_info.get('arch', config['model']['architecture'])
        
        # Временно подменяем архитектуру в конфиге
        cfg_copy = config.copy()
        cfg_copy['model'] = config['model'].copy()
        cfg_copy['model']['architecture'] = arch
        cfg_copy['model']['pretrained'] = False # Веса грузим из файла, imagenet не нужен
        
        try:
            # 2. Инициализация и загрузка весов
            model = create_model(cfg_copy)
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            
            # 3. Инференс
            y_true, y_pred, y_prob = evaluate_model(model, test_loader, device=device)
            
            # 4. Расчет метрик
            num_classes = cfg_copy['model'].get('num_classes', 2)
            if num_classes == 2:
                prob_pos = y_prob[:, 1]
                auc_val = roc_auc_score(y_true, prob_pos)
                f1 = f1_score(y_true, y_pred)
                
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                acc = accuracy_score(y_true, y_pred)
                
                # Данные для ROC кривой
                fpr, tpr, _ = roc_curve(y_true, prob_pos)
                roc_data[name] = (fpr, tpr, auc_val)
                
                # Клинические метрики
                sens_at_90_spec = tpr[np.where(fpr <= 0.10)[0][-1]] if len(np.where(fpr <= 0.10)[0]) > 0 else 0.0
                sens_at_95_spec = tpr[np.where(fpr <= 0.05)[0][-1]] if len(np.where(fpr <= 0.05)[0]) > 0 else 0.0
                missed_severe = int(np.sum((y_true == 1) & (y_pred == 0)))
                
                # Данные для Precision-Recall кривой
                precision, recall, _ = precision_recall_curve(y_true, prob_pos)
                pr_auc = auc(recall, precision)
                pr_data[name] = (recall, precision, pr_auc)
                
            else:
                # Фоллбэк для мультикласса
                auc_val = roc_auc_score(y_true, y_prob, multi_class='ovr')
                f1 = f1_score(y_true, y_pred, average='weighted')
                sens, spec = 0.0, 0.0
                acc = accuracy_score(y_true, y_pred)
                sens_at_90_spec = 0.0
                sens_at_95_spec = 0.0
                missed_severe = 0
                
            results.append({
                'Model': name,
                'Architecture': arch,
                'Accuracy': acc,
                'AUC': auc_val,
                'F1 Score': f1,
                'Sensitivity': sens,
                'Specificity': spec,
                'Sens @ 90% Spec': sens_at_90_spec,
                'Sens @ 95% Spec': sens_at_95_spec,
                'Missed Severe': missed_severe
            })
            
        except Exception as e:
            print(f"Ошибка при оценке {name}: {e}")
            
    # 5. Сохранение CSV и графиков
    df_results = pd.DataFrame(results)
    if len(df_results) > 0:
        df_results.to_csv(reports_dir / 'metrics_comparison.csv', index=False)
    
    if roc_data:
        plt.figure(figsize=(8, 6))
        for name, (fpr, tpr, auc_val) in roc_data.items():
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Сравнение ROC кривых')
        plt.legend(loc='lower right')
        plt.savefig(reports_dir / 'roc_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    if pr_data:
        plt.figure(figsize=(8, 6))
        for name, (recall, precision, pr_auc) in pr_data.items():
            plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Сравнение Precision-Recall кривых')
        plt.legend(loc='lower left')
        plt.savefig(reports_dir / 'pr_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    return df_results, reports_dir
