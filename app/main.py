import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

st.set_page_config(
    page_title="OptiX Medical", 
    page_icon="🏥", 
    layout="centered"
)

from app.utils_theme import apply_theme
apply_theme()

from src.utils.demo_generator import ensure_demo_data
# Генерация демо-данных при старте приложения
ensure_demo_data(project_root)

# Render logo
st.markdown("<h1 style='text-align: center; color: #0056b3;'>👁️ OptiX Medical</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #64748b;'>Система ИИ-помощи в диагностике ретинопатии недоношенных</h4>", unsafe_allow_html=True)

st.divider()

# Адаптивный блок, который выглядит неплохо и в светлой, и в тёмной теме
st.markdown("""
<div style='background-color: rgba(0, 86, 179, 0.1); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h3 style='color: #0056b3; margin-top: 0;'>О системе</h3>
    <p style='font-size: 16px;'>
        <strong>OptiX</strong> — это специализированный ассистент для офтальмологов, разработанный для анализа снимков глазного дна у недоношенных детей. 
        Система использует современные сверточные нейросети для автоматического выявления признаков тяжелых патологий и формирования визуальных отчетов (Grad-CAM).
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### Навигация (левое меню):
1. 📂 **Данные** — загрузка, проверка и подготовка медицинских снимков.
2. 🧠 **Обучение** — запуск и мониторинг тренировки ИИ моделей.
3. 🔬 **Анализ** — рабочее место врача: загрузка снимка пациента и получение предсказания.
4. 📊 **Результаты** — архив предыдущих анализов и отчеты.
5. ⚖️ **Сравнение моделей** — оценка моделей на тестовых данных.
6. ℹ️ **О проекте** — техническая документация и авторы.
7. 📖 **Инструкция** — руководство пользователя.
""")

st.info("👈 Пожалуйста, выберите нужный раздел в боковом меню для начала работы.")

st.divider()

st.subheader("🧪 Демо-режим (Для жюри и тестов)")
st.write("Если у вас нет реальных медицинских снимков для проверки системы на вкладке **Анализ**, вы можете скачать эти синтетические примеры:")

demo_dir = project_root / "demo_data"
col1, col2 = st.columns(2)
with col1:
    norm_path = demo_dir / "demo_normal.jpg"
    if norm_path.exists():
        with open(norm_path, "rb") as file:
            st.download_button("📥 Скачать: Норма (Здоровая сетчатка)", data=file, file_name="demo_normal.jpg", mime="image/jpeg", use_container_width=True)
            
with col2:
    rop_path = demo_dir / "demo_rop_stage3.jpg"
    if rop_path.exists():
        with open(rop_path, "rb") as file:
            st.download_button("📥 Скачать: ROP (Патология)", data=file, file_name="demo_rop_stage3.jpg", mime="image/jpeg", use_container_width=True)
