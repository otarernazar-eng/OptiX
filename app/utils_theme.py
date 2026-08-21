import streamlit as st

def apply_theme():
    """
    Инжектит CSS для тёмной или светлой темы 
    и управляет переключателем в сайдбаре.
    """
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
        
    st.sidebar.divider()
    st.sidebar.write("### Оформление интерфейса")
    dark_mode = st.sidebar.toggle("🌙 Тёмная тема", value=st.session_state.dark_mode)
    
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
        
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        .stApp {
            background-color: #121212;
            color: #E0E0E0;
        }
        .stApp [data-testid="stHeader"] {
            background-color: #121212;
        }
        .stApp [data-testid="stSidebar"] {
            background-color: #1E1E1E;
        }
        .stApp [data-testid="stMetricValue"] {
            color: #FFFFFF;
        }
        .stApp [data-testid="stMarkdownContainer"] p, 
        .stApp [data-testid="stMarkdownContainer"] li {
            color: #E0E0E0;
        }
        .stApp h1, .stApp h2, .stApp h3 {
            color: #FFFFFF;
        }
        div[data-baseweb="select"] > div {
            background-color: #2D2D2D;
            color: #FFFFFF;
        }
        </style>
        """, unsafe_allow_html=True)
