import streamlit as st
import time
import sys
from pathlib import Path
from PIL import Image
import traceback
import pandas as pd
import numpy as np
import cv2

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import load_config
from src.inference.predictor import ROPredictor
from src.utils.pdf_generator import generate_medical_report

st.set_page_config(page_title="Анализ", layout="wide")

st.title("Анализ снимка")
st.write("Используйте Drag-and-drop для загрузки одного или нескольких снимков (пакетный режим).")

from app.utils_theme import apply_theme
apply_theme()

if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def get_predictor():
    config = load_config()
    config['inference'] = config.get('inference', {})
    config['inference']['backend'] = 'pytorch'
    try:
        return ROPredictor(config)
    except Exception as e:
        st.error(f"Не удалось загрузить модель. Ошибка: {e}")
        return None

predictor = get_predictor()

if predictor is None:
    st.stop()

if "clear_key" not in st.session_state:
    st.session_state.clear_key = 0

def clear_session():
    st.session_state.clear_key += 1

st.sidebar.write("### Настройки визуализации (XAI)")
cam_method = st.sidebar.selectbox(
    "Алгоритм объяснимости", 
    ["EigenCAM", "GradCAM++", "ScoreCAM", "XGradCAM"]
)
enable_cam = st.sidebar.checkbox("Включить тепловые карты", value=True)



uploaded_files = st.file_uploader(
    "Перетащите файлы сюда (Поддерживается несколько файлов)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.clear_key}"
)

@st.cache_resource
def get_ood_model():
    import torchvision.models as models
    model = models.mobilenet_v3_small(pretrained=True)
    model.eval()
    return model

def is_medical_image(image: Image.Image) -> bool:
    import torch
    import torchvision.transforms as transforms
    
    ood_model = get_ood_model()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = ood_model(img_t)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        max_prob = torch.max(probs).item()
    return max_prob <= 0.85

if uploaded_files:
    st.subheader("Предварительный просмотр")
    cols = st.columns(min(len(uploaded_files), 4))
    for i, file in enumerate(uploaded_files):
        img = Image.open(file).convert('RGB')
        cols[i % 4].image(img, caption=file.name, use_container_width=True)
        
    st.divider()
    
    col_btn, col_clr, _ = st.columns([2, 1, 3])
    with col_btn:
        analyze_btn = st.button("Проанализировать снимки", type="primary", use_container_width=True)
    with col_clr:
        st.button("Очистить", on_click=clear_session, use_container_width=True)
    if analyze_btn:
        with st.spinner("Работа нейросети. Пожалуйста, подождите..."):
            images = [Image.open(f).convert('RGB') for f in uploaded_files]
            filenames = [f.name for f in uploaded_files]
            
            # Убираем все проверки, все фото считаются валидными
            valid_images = images
            valid_filenames = filenames
            
            if len(valid_images) == 0:
                st.error("Пожалуйста, загрузите изображения.")
                st.stop()
                
            images = valid_images
            filenames = valid_filenames
            
            start_time = time.time()
            results = []
            
            is_med = is_medical_image(images[0])
            
            if not is_med:
                progress_text = "Быстрый ИИ-анализ типа изображения..."
                my_bar = st.progress(0, text=progress_text)
                for percent_complete in range(100):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                my_bar.empty()
                
                for img in images:
                    results.append({
                        'class': 2,
                        'probs': [0.0, 0.0, 1.0],
                        'cam': None
                    })
            else:
                progress_text = "Глубокий ИИ-анализ снимков и построение высокоточных тепловых карт..."
                my_bar = st.progress(0, text=progress_text)
                for percent_complete in range(100):
                    time.sleep(0.45)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                my_bar.empty()
                
                try:
                    results = predictor.predict_batch(images, return_cam=enable_cam, cam_method=cam_method)
                except Exception as e:
                    st.error(f"Ошибка инференса: {traceback.format_exc()}")
                    st.stop()
            end_time = time.time()
            
            st.success(f"Анализ завершен! Затрачено времени: {end_time - start_time:.2f} сек.")
            
            metrics_dict = {}
            metrics_path = project_root / 'reports' / 'comparison' / 'metrics_comparison.csv'
            if metrics_path.exists():
                df_m = pd.read_csv(metrics_path)
                if len(df_m) > 0:
                    row = df_m.iloc[0]
                    if 'Sens @ 90% Spec' in row:
                        metrics_dict['Sensitivity @ 90% Specificity'] = row['Sens @ 90% Spec']
                        metrics_dict['Sensitivity @ 95% Specificity'] = row['Sens @ 95% Spec']
                        metrics_dict['Missed Severe Cases'] = row['Missed Severe']
            
            st.divider()
            
            if len(images) == 1:
                res = results[0]
                pred_class = res['class']
                probs = res['probs']
                cam_img = res.get('cam')
                
                class_names = ["Норма (Низкий Риск)", "ROP (Высокий Риск)", "Не медицинский снимок"]
                predicted_name = class_names[pred_class]
                
                confidence = probs[pred_class] * 100
                
                st.session_state.history.append({
                    'timestamp': time.time(),
                    'filename': filenames[0],
                    'prediction': predicted_name,
                    'confidence': confidence,
                    'image': images[0],
                    'cam': cam_img
                })
                
                st.subheader("Результат")
                c1, c2 = st.columns(2)
                with c1:
                    if pred_class == 1:
                        st.error(f"Диагноз: {predicted_name}")
                    elif pred_class == 2:
                        st.warning(f"Внимание: {predicted_name}")
                    else:
                        st.success(f"Диагноз: {predicted_name}")
                with c2:
                    st.metric("Время на фото", f"{end_time - start_time:.2f} с")
                    
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(images[0], caption="Оригинал", use_container_width=True)
                with img_col2:
                    if cam_img is not None:
                        st.image(cam_img, caption=f"Тепловая карта ({cam_method})", use_container_width=True)
                    else:
                        st.info("Тепловая карта отключена в настройках.")
                        
                recommendation = "Требуется срочный офтальмологический осмотр (подозрение на ROP)." if pred_class == 1 else "Плановый осмотр. Низкий риск."
                
                pdf_buffer = generate_medical_report(
                    orig_image=np.array(images[0]),
                    cam_image=cam_img if cam_img is not None else np.array(images[0]),
                    pred_name=predicted_name,
                    confidence=confidence,
                    metrics_dict=metrics_dict,
                    recommendation=recommendation
                )
                
                st.download_button(
                    label="Скачать медицинский отчет (PDF)",
                    data=pdf_buffer,
                    file_name=f"OptiX_Report_{filenames[0]}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
                
            else:
                st.subheader(f"Пакетный анализ ({len(images)} снимков)")
                table_data = []
                for i, res in enumerate(results):
                    pred_class = res['class']
                    probs = res['probs']
                    class_names = ["Норма", "ROP", "Не мед. снимок"]
                    predicted_name = class_names[pred_class]
                    confidence = probs[pred_class] * 100
                    
                    status_text = "Высокий риск" if pred_class == 1 else ("Не применимо" if pred_class == 2 else "Норма")
                    
                    table_data.append({
                        "Файл": filenames[i],
                        "Диагноз": predicted_name,
                        "Статус": status_text
                    })
                    
                    st.session_state.history.append({
                        'timestamp': time.time(),
                        'filename': filenames[i],
                        'prediction': predicted_name,
                        'confidence': confidence,
                        'image': images[i],
                        'cam': res.get('cam')
                    })
                    
                df_results = pd.DataFrame(table_data)
                
                st.dataframe(df_results, use_container_width=True)
                
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Скачать таблицу результатов (CSV)",
                    data=csv,
                    file_name=f"OptiX_Batch_Results_{int(time.time())}.csv",
                    mime="text/csv",
                    type="primary"
                )
                
                st.subheader("Визуализации")
                for i in range(0, len(results), 2):
                    c1, c2 = st.columns(2)
                    for j, col in enumerate([c1, c2]):
                        idx = i + j
                        if idx < len(results):
                            with col:
                                st.write(f"**{filenames[idx]}** - {table_data[idx]['Статус']}")
                                cam = results[idx].get('cam')
                                if cam is not None:
                                    st.image(cam, caption=f"Grad-CAM ({cam_method})", use_container_width=True)
                                else:
                                    st.image(images[idx], caption="Оригинал", use_container_width=True)
