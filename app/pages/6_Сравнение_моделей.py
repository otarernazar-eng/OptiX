import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Подключаем корень
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import load_config
from src.evaluation.compare import compare_models

st.set_page_config(page_title="Сравнение моделей", page_icon="⚖️", layout="wide")

from app.utils_theme import apply_theme
apply_theme()

st.title("⚖️ Сравнение моделей")
st.write("Выберите сохраненные веса (`.pth`) из архива для проведения независимого тестирования на **отложенной (test)** выборке.")

models_dir = project_root / 'models'
available_models = list(models_dir.glob('*.pth')) if models_dir.exists() else []

if not available_models:
    st.warning("В папке `models/` нет сохраненных моделей (.pth). Сначала обучите хотя бы одну модель на странице 'Обучение'.")
    st.stop()

st.subheader("1. Выбор моделей")

arch_options = ["efficientnet-b0", "efficientnet-b2", "resnet50", "mobilenetv3-large", "convnext-tiny"]
models_to_compare = []

# Формируем UI таблицу для выбора
for i, model_path in enumerate(available_models):
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        use_model = st.checkbox("Выбрать", key=f"use_{i}")
    with col2:
        st.write(f"**{model_path.name}**")
        name = st.text_input("Название (для легенды графиков)", value=model_path.stem, key=f"name_{i}")
    with col3:
        # Пытаемся угадать архитектуру из названия файла
        default_arch = "efficientnet-b0"
        for a in arch_options:
            if a in model_path.name.lower():
                default_arch = a
                break
        arch = st.selectbox("Архитектура", arch_options, index=arch_options.index(default_arch), key=f"arch_{i}")
        
    if use_model:
        models_to_compare.append({
            'name': name,
            'path': str(model_path),
            'arch': arch
        })

st.divider()

# Кнопка запуска
if st.button("🚀 Запустить анализ", type="primary", disabled=len(models_to_compare) == 0):
    if len(models_to_compare) < 1:
        st.error("Выберите хотя бы одну модель!")
    else:
        with st.spinner("Прогон моделей через тестовый датасет... Это может занять некоторое время."):
            config = load_config()
            try:
                df_results, reports_dir = compare_models(config, models_to_compare)
                st.session_state['compare_results_df'] = df_results
                st.session_state['compare_reports_dir'] = reports_dir
                st.success("Сравнение успешно завершено!")
            except Exception as e:
                st.error(f"Ошибка при инференсе: {e}")

# Отрисовка результатов
if 'compare_results_df' in st.session_state:
    st.subheader("2. Сводная таблица метрик (Тестовая выборка)")
    df = st.session_state['compare_results_df']
    
    # Подсвечиваем максимальные значения по каждой колонке зелёным
    styled_df = df.style.highlight_max(
        subset=['Accuracy', 'AUC', 'F1 Score', 'Sensitivity', 'Specificity'], 
        color='#d4edda', 
        axis=0
    ).format({
        'Accuracy': '{:.4f}',
        'AUC': '{:.4f}',
        'F1 Score': '{:.4f}',
        'Sensitivity': '{:.4f}',
        'Specificity': '{:.4f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    st.subheader("3. Визуализация")
    reports_dir = st.session_state['compare_reports_dir']
    
    col1, col2 = st.columns(2)
    with col1:
        roc_path = reports_dir / 'roc_comparison.png'
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC-кривые")
            
    with col2:
        pr_path = reports_dir / 'pr_comparison.png'
        if pr_path.exists():
            st.image(str(pr_path), caption="Precision-Recall кривые")
