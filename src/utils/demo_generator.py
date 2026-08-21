import cv2
import numpy as np
from pathlib import Path

def generate_synthetic_retina(stage='normal', size=512):
    """
    Генерирует синтетическое изображение глазного дна (сетчатки)
    с использованием простых геометрических паттернов.
    Используется для демо-режима на защите, если реальных данных нет.
    """
    # Базовый фон (оранжево-красный)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = [30, 70, 180] # BGR
    
    # Добавляем радиальный градиент (затемнение по краям)
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - size/2)**2 + (j - size/2)**2)
            if dist > size/2.2:
                img[i, j] = [0, 0, 0] # черная маска по краям
            else:
                intensity = 1.0 - (dist / (size/2)) * 0.5
                img[i, j] = (img[i, j] * intensity).astype(np.uint8)

    # Диск зрительного нерва (светло-желтый круг)
    center_od = (int(size * 0.75), int(size * 0.5))
    cv2.circle(img, center_od, int(size * 0.12), (120, 200, 230), -1)
    
    # Кровеносные сосуды
    np.random.seed(42 if stage == 'normal' else np.random.randint(0, 1000))
    
    num_vessels = 12 if stage == 'normal' else 20
    for _ in range(num_vessels):
        x, y = center_od
        angle = np.random.uniform(0, 2 * np.pi)
        
        thickness = np.random.randint(3, 6)
        for step in range(120):
            if stage == 'normal':
                # Прямые гладкие сосуды
                angle += np.random.uniform(-0.05, 0.05)
            else:
                # Извитые сосуды (Tortuosity) - признак ROP плюс-болезни
                angle += np.random.uniform(-0.35, 0.35)
                
            length = np.random.randint(5, 15)
            nx = int(x + np.cos(angle) * length)
            ny = int(y + np.sin(angle) * length)
            
            if 0 <= nx < size and 0 <= ny < size:
                cv2.line(img, (x, y), (nx, ny), (10, 20, 100), thickness)
            x, y = nx, ny
            # Сосуды сужаются к периферии
            thickness = max(1, thickness - (1 if step % 25 == 0 else 0))

    if stage != 'normal':
        # Добавляем кровоизлияния (hemorrhages) или экссудаты
        for _ in range(15):
            rx = np.random.randint(int(size*0.2), int(size*0.8))
            ry = np.random.randint(int(size*0.2), int(size*0.8))
            cv2.circle(img, (rx, ry), np.random.randint(2, 8), (15, 25, 130), -1)

    # Добавляем шум камеры для реалистичности
    noise = np.random.normal(0, 8, img.shape).astype(np.int8)
    img = cv2.add(img, noise, dtype=cv2.CV_8UC3)
    
    # Сглаживание
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img

def ensure_demo_data(project_root: Path):
    demo_dir = project_root / "demo_data"
    demo_dir.mkdir(exist_ok=True)
    
    # Если файлы уже есть, не перегенерируем
    if len(list(demo_dir.glob("*.jpg"))) >= 2:
        return
        
    print("Генерация синтетических демо-изображений...")
    
    # Норма
    norm = generate_synthetic_retina('normal')
    cv2.imwrite(str(demo_dir / "demo_normal.jpg"), norm)
    
    # Патология
    rop = generate_synthetic_retina('rop')
    cv2.imwrite(str(demo_dir / "demo_rop_stage3.jpg"), rop)
