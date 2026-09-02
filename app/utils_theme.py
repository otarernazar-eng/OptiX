import streamlit as st

def apply_theme():
    """
    Инжектит CSS для темно-синей темы без переключателей.
    """
    st.markdown("""
    <style>
    .stApp {
        background-color: #0A192F;
        color: #E0E0E0;
    }
    .stApp [data-testid="stHeader"] {
        background-color: #0A192F;
    }
    .stApp [data-testid="stSidebar"] {
        background-color: #061020;
    }
    .stApp [data-testid="stMetricValue"] {
        color: #FFFFFF;
    }
    .stApp [data-testid="stMarkdownContainer"] p, 
    .stApp [data-testid="stMarkdownContainer"] li {
        color: #CBD5E1;
    }
    .stApp h1, .stApp h2, .stApp h3 {
        color: #60A5FA;
    }
    div[data-baseweb="select"] > div {
        background-color: #1E293B;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
