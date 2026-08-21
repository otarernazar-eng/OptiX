import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import time

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import load_config
from src.data.loader import prepare_data
from src.data.preprocess import prepare_datasets

st.set_page_config(page_title="Данные", page_icon="📂", layout="wide")

from app.utils_theme import apply_theme
apply_theme()

config = load_config()
raw_dir = project_root / config['data'].get('raw_dir', 'data/raw')
metadata_path = raw_dir / "metadata.csv"

st.title("📂 Управление медицинскими данными")
st.markdown("Подготовка наборов снимков (i-ROP, RIDIRP, Macretina) к обучению.")

# Check status
is_loaded = metadata_path.exists()
try:
    if is_loaded:
        df = pd.read_csv(metadata_path)
        is_loaded = len(df) > 0
    else:
        df = None
except Exception:
    is_loaded = False
    df = None

# Show Status Blocks
col1, col2, col3 = st.columns(3)
with col1:
    if is_loaded:
        st.success("Статус: Загружены и готовы")
    else:
        st.error("Статус: Данные отсутствуют")
with col2:
    st.info(f"Снимков: {len(df) if is_loaded else 0}")
with col3:
    if is_loaded:
        dist = df['label'].value_counts().to_dict()
        dist_str = " | ".join([f"{k}: {v}" for k, v in dist.items()])
        st.info(f"Распределение: {dist_str}")
    else:
        st.info("Распределение: N/A")

st.divider()

if st.button("🔄 Подготовить / Обновить данные", type="primary"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def ui_progress_callback(message, fraction):
        status_text.text(message)
        # Allocate 0 to 0.7 for the loader part
        progress_bar.progress(fraction * 0.7)
        time.sleep(0.3)
        
    try:
        # Step 1: Loading
        csv_path = prepare_data(config, progress_callback=ui_progress_callback)
        
        # Step 2: Preprocess, Splits, EDA
        status_text.text("Разбиение датасета и генерация графиков (EDA)...")
        progress_bar.progress(0.85)
        prepare_datasets(config)
        
        status_text.text("Готово!")
        progress_bar.progress(1.0)
        
        st.success(f"Данные успешно подготовлены!")
        st.balloons()
        time.sleep(2)
        st.rerun()
    except Exception as e:
        st.error(f"Ошибка при подготовке: {e}")

if is_loaded:
    st.subheader("Превью базы данных")
    st.dataframe(df.head(10), use_container_width=True)
    
    eda_path = project_root / 'reports' / 'eda' / 'class_distribution.png'
    aug_path = project_root / 'reports' / 'aug_examples.png'
    
    col_a, col_b = st.columns(2)
    with col_a:
        if eda_path.exists():
            st.subheader("Разбиение классов")
            st.image(str(eda_path), use_column_width=True)
            
    with col_b:
        if aug_path.exists():
            st.subheader("Примеры аугментаций")
            st.image(str(aug_path), use_column_width=True)
