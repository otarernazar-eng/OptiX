import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.factory import create_model
from src.data.augmentations import get_albumentations_transforms
from src.inference.explain import generate_heatmap

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

class ROPredictor:
    """
    Класс инференса для моделей ROP (Retinopathy).
    Поддерживает PyTorch и ONNX бэкенды.
    Реализует кэширование модели в памяти (Паттерн Singleton) 
    и генерацию продвинутых тепловых карт (через src/inference/explain.py).
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ROPredictor, cls).__new__(cls)
        return cls._instance

    def __init__(self, config, model_path=None):
        if hasattr(self, 'initialized') and self.initialized:
            if model_path is None or self.model_path == model_path:
                return
            
        self.config = config
        self.inference_cfg = config.get('inference', {})
        self.backend = self.inference_cfg.get('backend', 'pytorch').lower()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model_path = model_path or self.inference_cfg.get('model_path', 'models/best.pth')
        
        image_size = config['training'].get('image_size', 384)
        _, self.transform = get_albumentations_transforms(image_size)
        
        if self.backend == 'onnx':
            if not HAS_ONNX:
                raise ImportError("ONNX Runtime не установлен. Запустите: pip install onnxruntime")
            self._load_onnx()
        else:
            self._load_pytorch()
            
        self.initialized = True
        
    def _load_onnx(self):
        print(f"Загрузка ONNX модели из {self.model_path}")
        providers = ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.ort_session.get_inputs()[0].name
        
    def _load_pytorch(self):
        print(f"Загрузка PyTorch модели из {self.model_path}")
        cfg_copy = self.config.copy()
        if 'model' not in cfg_copy:
            cfg_copy['model'] = {}
        cfg_copy['model']['pretrained'] = False
        
        self.model = create_model(cfg_copy)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        elif isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        transformed = self.transform(image=image)['image']
        
        if isinstance(transformed, torch.Tensor):
            tensor = transformed.unsqueeze(0)
        else:
            tensor = torch.tensor(transformed).unsqueeze(0)
            
        return tensor, image

    def predict(self, image, return_cam=True, cam_method='GradCAM'):
        """Инференс для одного изображения."""
        return self.predict_batch([image], return_cam=return_cam, cam_method=cam_method)[0]

    def predict_batch(self, images, return_cam=True, cam_method='GradCAM'):
        """Батч-инференс для списка изображений."""
        tensors = []
        orig_images = []
        
        for img in images:
            t, o = self._preprocess(img)
            tensors.append(t)
            orig_images.append(o)
            
        batch_tensor = torch.cat(tensors, dim=0)
        
        if self.backend == 'onnx':
            input_data = batch_tensor.numpy()
            ort_inputs = {self.input_name: input_data}
            logits = self.ort_session.run(None, ort_inputs)[0]
            logits = torch.tensor(logits)
        else:
            batch_tensor = batch_tensor.to(self.device)
            with torch.no_grad():
                logits = self.model(batch_tensor)
                
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        
        results = []
        arch_name = self.config['model'].get('architecture', '')
        
        for i in range(len(images)):
            res = {
                'class': int(preds[i]),
                'probs': probs[i].tolist()
            }
            
            # Интеграция с модулем Explainability
            if return_cam and self.backend == 'pytorch':
                input_tensor = batch_tensor[i:i+1] # Берем тензор с размерностью батча 1 (сохраняя девайс)
                orig_img = orig_images[i]
                
                cam_img = generate_heatmap(
                    model=self.model,
                    input_tensor=input_tensor,
                    orig_image=orig_img,
                    arch_name=arch_name,
                    method=cam_method
                )
                res['cam'] = cam_img
                
            results.append(res)
            
        return results
