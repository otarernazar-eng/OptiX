import streamlit as st
import threading
import time
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
from pathlib import Path
import traceback

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx
except ImportError:
    from streamlit.runtime.scriptrunner.script_run_context import add_script_run_ctx

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import load_config
from src.data.preprocess import prepare_datasets
from src.data.dataloader import get_dataloaders
from src.models.factory import create_model
from src.train.trainer import Trainer

st.set_page_config(page_title="Обучение", page_icon="🧠", layout="wide")

from app.utils_theme import apply_theme
apply_theme()

st.title("🧠 Запуск обучения модели")

# Session State Initialization
if 'is_training' not in st.session_state:
    st.session_state.is_training = False
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = []
if 'best_model_path' not in st.session_state:
    st.session_state.best_model_path = None
if 'train_error' not in st.session_state:
    st.session_state.train_error = None
if 'model' not in st.session_state:
    st.session_state.model = None

config = load_config()

# UI Sidebar for Hyperparameters
with st.sidebar:
    st.header("Настройки обучения")
    arch_options = ["efficientnet-b0", "efficientnet-b2", "resnet50", "mobilenetv3-large", "convnext-tiny"]
    default_arch = config['model'].get('architecture', 'efficientnet-b0').lower()
    if default_arch not in arch_options:
        default_arch = "efficientnet-b0"
        
    arch = st.selectbox("Архитектура", arch_options, index=arch_options.index(default_arch))
    epochs = st.number_input("Эпохи", min_value=1, max_value=200, value=config['training'].get('epochs', 50))
    batch_size = st.number_input("Размер батча", min_value=1, max_value=128, value=config['training'].get('batch_size', 16))
    
    loss_options = ["ce", "focal", "weighted_ce"]
    default_loss = config['training'].get('loss_type', 'focal').lower()
    if default_loss not in loss_options:
        default_loss = "focal"
        
    loss_type = st.selectbox("Лосс (Функция потерь)", loss_options, index=loss_options.index(default_loss))
    freeze_bb = st.checkbox("Заморозить backbone (обучать только голову)", value=config['model'].get('freeze_backbone', False))

def training_thread_func(run_config):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Data preparation
        train_ds, val_ds, test_ds = prepare_datasets(run_config)
        train_loader, val_loader, test_loader = get_dataloaders(run_config, train_ds, val_ds, test_ds)
        
        # Model
        model = create_model(run_config)
        
        # Optimization
        lr = run_config['training'].get('learning_rate', 0.001)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Callback for real-time Streamlit updates
        def epoch_callback(epoch, metrics):
            metrics_copy = metrics.copy()
            metrics_copy['epoch'] = epoch
            st.session_state.metrics_history.append(metrics_copy)
            
        trainer = Trainer(
            config=run_config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch_callback=epoch_callback
        )
        
        result = trainer.fit()
        
        st.session_state.best_model_path = result['model_path']
        st.session_state.model = trainer.model # Keep in memory for ONNX export
        st.session_state.training_complete = True
    except Exception as e:
        st.session_state.train_error = traceback.format_exc()
    finally:
        st.session_state.is_training = False

# Main Area Layout
plot_placeholder = st.empty()
progress_placeholder = st.empty()

start_col, export_col = st.columns(2)

with start_col:
    if st.button("🚀 Начать обучение", disabled=st.session_state.is_training, type="primary"):
        # Reset previous states
        st.session_state.metrics_history = []
        st.session_state.training_complete = False
        st.session_state.train_error = None
        st.session_state.is_training = True
        
        # Prepare runtime config
        run_config = load_config()
        run_config['model']['architecture'] = arch
        run_config['model']['freeze_backbone'] = freeze_bb
        run_config['training']['epochs'] = epochs
        run_config['training']['batch_size'] = batch_size
        run_config['training']['loss_type'] = loss_type
        
        # Start background thread
        t = threading.Thread(target=training_thread_func, args=(run_config,))
        add_script_run_ctx(t) # Streamlit magic to allow session_state access from thread
        t.start()
        
        st.rerun()

# --- Dynamic Real-Time Updates ---
if st.session_state.is_training or st.session_state.training_complete:
    history = st.session_state.metrics_history
    
    if len(history) > 0:
        df_metrics = pd.DataFrame(history)
        
        with plot_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Функция потерь (Loss)")
                if 'train_loss' in df_metrics.columns and 'val_loss' in df_metrics.columns:
                    st.line_chart(df_metrics.set_index('epoch')[['train_loss', 'val_loss']])
            with col2:
                st.subheader("Качество (AUC)")
                if 'val_auc' in df_metrics.columns:
                    st.line_chart(df_metrics.set_index('epoch')[['val_auc']])
        
        if st.session_state.is_training:
            current_epoch = history[-1]['epoch']
            progress_placeholder.progress(current_epoch / epochs)
            progress_placeholder.write(f"⏳ **Обучение в процессе:** Эпоха {current_epoch} из {epochs}")
    
    # Auto-refresh loop while training
    if st.session_state.is_training:
        time.sleep(2)
        st.rerun()

# --- Post-Training Status ---
if st.session_state.train_error:
    st.error(f"Произошла ошибка при обучении:\n```python\n{st.session_state.train_error}\n```")

if st.session_state.training_complete:
    progress_placeholder.success(f"✅ Обучение успешно завершено! Лучшие веса сохранены: `{st.session_state.best_model_path}`")
    
    st.divider()
    st.subheader("🏆 Лучшие метрики валидации")
    if len(st.session_state.metrics_history) > 0:
        best_epoch = max(st.session_state.metrics_history, key=lambda x: x['val_auc'])
        best_df = pd.DataFrame([best_epoch])
        st.dataframe(best_df, use_container_width=True)
        
    with export_col:
        if st.button("💾 Сохранить в ONNX", type="primary"):
            with st.spinner("Экспорт модели в ONNX..."):
                try:
                    export_path = Path(st.session_state.best_model_path).with_suffix('.onnx')
                    
                    model = st.session_state.model
                    model.eval()
                    device = next(model.parameters()).device
                    
                    img_size = config['training'].get('image_size', 384)
                    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
                    
                    torch.onnx.export(
                        model, 
                        dummy_input, 
                        str(export_path),
                        export_params=True,
                        opset_version=14,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                    )
                    st.success(f"Модель успешно конвертирована! Файл доступен по пути: `{export_path}`")
                except Exception as e:
                    st.error(f"Ошибка конвертации ONNX: {e}")
