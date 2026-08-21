import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_target_layer(model, arch_name):
    """
    Пытается автоматически определить последний сверточный слой 
    нейросети на основе её названия (для Grad-CAM).
    """
    arch = arch_name.lower()
    if 'efficientnet' in arch:
        return [model.conv_head]
    elif 'resnet' in arch:
        return [model.layer4[-1]]
    elif 'convnext' in arch:
        return [model.stages[-1].blocks[-1]]
    else:
        # Универсальный fallback
        return [list(model.children())[-2]]

def generate_heatmap(model, input_tensor, orig_image, arch_name, method='GradCAM', target_layer=None):
    """
    Генерирует тепловую карту и накладывает её на оригинальное изображение.
    
    Args:
        model: PyTorch модель
        input_tensor: тензор [1, C, H, W]
        orig_image: исходное изображение (numpy array RGB)
        arch_name: название архитектуры
        method: алгоритм (GradCAM, GradCAM++, ScoreCAM и т.д.)
        target_layer: целевой слой для визуализации
        
    Returns:
        numpy array изображения с наложенной тепловой картой
    """
    if target_layer is None:
        target_layer = get_target_layer(model, arch_name)
        
    cam_classes = {
        'GradCAM': GradCAM,
        'GradCAM++': GradCAMPlusPlus,
        'ScoreCAM': ScoreCAM,
        'XGradCAM': XGradCAM,
        'EigenCAM': EigenCAM
    }
    
    CamConstructor = cam_classes.get(method, GradCAM)
    
    device = next(model.parameters()).device
    use_cuda = device.type == 'cuda'
    
    try:
        with CamConstructor(model=model, target_layers=target_layer, use_cuda=use_cuda) as cam:
            # Генерация маски (grayscale)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
            
            # Приводим оригинальное изображение к размеру тензора, чтобы наложить маску
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            orig_resized = cv2.resize(orig_image, (w, h))
            orig_float = orig_resized.astype(np.float32) / 255.0
            
            # Наложение
            visualization = show_cam_on_image(orig_float, grayscale_cam, use_rgb=True)
            return visualization
    except Exception as e:
        print(f"Ошибка при генерации {method}: {e}")
        return None
