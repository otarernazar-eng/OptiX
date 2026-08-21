from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import time
from PIL import Image

def generate_medical_report(orig_image, cam_image, pred_name, confidence, metrics_dict, recommendation):
    """
    Генерирует PDF отчет с использованием ReportLab (на английском языке для совместимости шрифтов).
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Заголовок
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "OptiX Medical AI - Clinical Report")
    
    # Метаданные
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 90, f"Date generated: {time.strftime('%Y-%m-%d %H:%M')}")
    c.drawString(50, height - 110, f"AI Predicted Class: {pred_name}")
    c.drawString(50, height - 130, f"AI Confidence Score: {confidence:.2f}%")
    
    # Рекомендация
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 160, "Clinical Recommendation:")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 180, recommendation)
    
    # Клинические метрики (Clinical Utility Metrics)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 220, "Model Clinical Utility Metrics (Test Set Reference):")
    c.setFont("Helvetica", 12)
    y_offset = 240
    
    if metrics_dict:
        for k, v in metrics_dict.items():
            if isinstance(v, float):
                c.drawString(50, height - y_offset, f"- {k}: {v:.4f}")
            else:
                c.drawString(50, height - y_offset, f"- {k}: {v}")
            y_offset += 20
    else:
        c.drawString(50, height - y_offset, "No historical comparison metrics found.")
        y_offset += 20
        
    # Блок с изображениями
    y_images = height - y_offset - 250
    
    # Конвертация Numpy в PIL для ReportLab
    if not isinstance(orig_image, Image.Image):
        orig_img_pil = Image.fromarray(orig_image)
    else:
        orig_img_pil = orig_image
        
    if not isinstance(cam_image, Image.Image):
        cam_img_pil = Image.fromarray(cam_image)
    else:
        cam_img_pil = cam_image
        
    orig_reader = ImageReader(orig_img_pil)
    cam_reader = ImageReader(cam_img_pil)
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_images + 220, "Original Retinal Image:")
    c.drawImage(orig_reader, 50, y_images, width=200, height=200, preserveAspectRatio=True)
    
    c.drawString(300, y_images + 220, "AI Attention Map (Grad-CAM):")
    c.drawImage(cam_reader, 300, y_images, width=200, height=200, preserveAspectRatio=True)
    
    # Подвал
    c.setFont("Helvetica", 9)
    c.drawString(50, 30, "OptiX System - Decision Support Only. Not a substitute for professional medical diagnosis.")
    
    c.showPage()
    c.save()
    
    buffer.seek(0)
    return buffer
