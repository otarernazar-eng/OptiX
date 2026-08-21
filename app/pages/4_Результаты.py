import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parents[2]
import sys
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

st.set_page_config(page_title="История и Результаты", page_icon="📋", layout="wide")

from app.utils_theme import apply_theme
apply_theme()

st.title("📋 История анализов")
st.write("Все проведенные вами анализы в рамках текущей сессии сохраняются здесь.")

if 'history' not in st.session_state or len(st.session_state.history) == 0:
    st.info("История пуста. Перейдите на страницу 'Анализ' и загрузите снимки.")
else:
    if st.button("🗑️ Очистить всю историю", type="primary"):
        st.session_state.history = []
        st.rerun()
        
    st.divider()
    
    # Отображаем историю
    for idx, item in enumerate(reversed(st.session_state.history)):
        real_idx = len(st.session_state.history) - 1 - idx
        dt = datetime.fromtimestamp(item['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([1.5, 1.5, 2, 2, 1])
            with c1:
                st.image(item['image'], use_column_width=True, caption="Оригинал")
            with c2:
                if item.get('cam') is not None:
                    st.image(item['cam'], use_column_width=True, caption="Grad-CAM")
                else:
                    st.write("Тепловой карты нет")
            with c3:
                st.write(f"**Файл:** {item['filename']}")
                st.write(f"**Дата:** {dt}")
            with c4:
                # Цветовая кодировка
                if "Высокий" in item['prediction'] or "ROP" in item['prediction']:
                    st.error(f"**Диагноз:** {item['prediction']}")
                else:
                    st.success(f"**Диагноз:** {item['prediction']}")
                st.write(f"**Уверенность ИИ:** {item['confidence']:.1f}%")
            with c5:
                if st.button("Удалить", key=f"del_{real_idx}"):
                    st.session_state.history.pop(real_idx)
                    st.rerun()
        st.divider()

st.subheader("🏥 Оценка Клинической Полезности алгоритмов")
metrics_path = project_root / 'reports' / 'comparison' / 'metrics_comparison.csv'

if metrics_path.exists():
    df_metrics = pd.read_csv(metrics_path)
    
    clinical_cols = ['Model', 'Architecture', 'Sens @ 90% Spec', 'Sens @ 95% Spec', 'Missed Severe']
    if all(col in df_metrics.columns for col in clinical_cols):
        df_clinical = df_metrics[clinical_cols].copy()
        
        st.write("Метрики безопасности (из отложенной тестовой выборки):")
        
        styled_df = df_clinical.style.highlight_max(
            subset=['Sens @ 90% Spec', 'Sens @ 95% Spec'], 
            color='#3D9970', 
            axis=0
        ).highlight_min(
            subset=['Missed Severe'], 
            color='#3D9970', 
            axis=0
        ).format({
            'Sens @ 90% Spec': '{:.4f}',
            'Sens @ 95% Spec': '{:.4f}'
        })
        
        st.dataframe(styled_df, use_container_width=True)
        
        st.info("💡 **Sens @ 90% Spec**: Чувствительность при лимите в 10% ложных тревог.\n\n"
                "💡 **Sens @ 95% Spec**: Чувствительность при строгом лимите в 5% ложных тревог.\n\n"
                "💡 **Missed Severe**: Количество ложноотрицательных случаев (очень опасно).")
