import streamlit as st
import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

st.set_page_config(
    page_title="OptiX Medical", 
    layout="centered"
)

from app.utils_theme import apply_theme
apply_theme()

st.markdown("<h1 style='text-align: center; color: #60A5FA;'>OptiX Medical</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #CBD5E1;'>Система помощи в диагностике ретинопатии недоношенных</h4>", unsafe_allow_html=True)

st.divider()

st.header("О проекте")
st.write("""
OptiX — это специализированный ассистент для офтальмологов, разработанный для анализа снимков глазного дна у недоношенных детей. 
Система использует современные сверточные нейросети для автоматического выявления признаков тяжелых патологий и формирования визуальных отчетов.
""")

st.header("Проблема")
st.write("""
Ретинопатия недоношенных (ROP) — одна из ведущих причин предотвратимой детской слепоты во всем мире. 
В Казахстане, с развитием неонатологии и повышением выживаемости глубоко недоношенных детей, частота ROP неуклонно растет.
Своевременный скрининг в первые недели жизни позволяет предотвратить необратимую потерю зрения в 90% случаев.
Однако из-за нехватки узкоспециализированных детских офтальмологов в отдаленных регионах, ранняя диагностика сильно затруднена. 

OptiX Medical призван решить эту проблему, предоставляя врачам первичного звена ассистента для мгновенного "второго мнения" при скрининге.
""")

graph_path = os.path.join(project_root, "graph.png")
if os.path.exists(graph_path):
    st.image(graph_path, use_container_width=True, caption="Динамика выявленных случаев ретинопатии")
