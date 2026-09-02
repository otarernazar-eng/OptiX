import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from app.utils_theme import apply_theme

st.set_page_config(page_title="Данные", layout="wide")
apply_theme()

st.title("Данные для обучения")

st.write("В данном разделе представлены датасеты, фотографии из которых были использованы для обучения и тестирования нашей модели.")

st.header("1. RIDIRP (Чехия)")
st.write("""
- **Размер**: 6 004 изображения, 188 новорождённых
- **Лицензия**: Бесплатно для некоммерческого использования
- **Ссылка**: [https://doi.org/10.6084/m9.figshare.c.6626162.v1](https://doi.org/10.6084/m9.figshare.c.6626162.v1)
- **Источник**: University Hospital Ostrava
""")

st.divider()

st.header("2. Macretina (Индия)")
st.write("""
- **Размер**: 1 432 изображения, 112 недоношенных детей
- **Лицензия**: Публичный датасет на Figshare
- **Ссылка**: [https://figshare.com](https://figshare.com) (искать по названию Macretina)
- **Три поднабора**: Ridge (465), OD (500), BV (38)
""")

st.divider()

st.header("3. Macau University of Science and Technology (MUST)")
st.write("""
- **Описание**: 1 099 изображений от 483 младенцев. Классы: Normal, Stage 1-3 ROP, Laser scars.
- **Ссылка**: [https://figshare.com/articles/figure/25514449](https://figshare.com/articles/figure/25514449)
- **Статья**: [https://link.springer.com/article/10.1038/s41597-024-03433-7](https://link.springer.com/article/10.1038/s41597-024-03433-7)
""")
