import timm
import torch
import torch.nn as nn

def create_model(config):
    """
    Создает модель компьютерного зрения через timm по конфигурации.
    Поддерживаемые архитектуры: EfficientNet-B0/B2, ResNet50, MobileNetV3-Large, ConvNeXt-Tiny.
    Позволяет заморозить бэкбоун и настроить классификатор.
    """
    model_cfg = config.get('model', {})
    
    arch_name = model_cfg.get('architecture', 'efficientnet-b0').lower()
    
    # Маппинг удобных названий в названия, которые ожидает библиотека timm
    arch_mapping = {
        'efficientnet-b0': 'efficientnet_b0',
        'efficientnet-b2': 'efficientnet_b2',
        'resnet50': 'resnet50',
        'mobilenetv3-large': 'mobilenetv3_large_100',
        'convnext-tiny': 'convnext_tiny'
    }
    
    timm_arch = arch_mapping.get(arch_name, arch_name)
    
    num_classes = model_cfg.get('num_classes', 2)
    pretrained = model_cfg.get('pretrained', True)
    freeze_backbone = model_cfg.get('freeze_backbone', False)
    drop_rate = model_cfg.get('drop_rate', 0.2)
    
    try:
        # timm автоматически настраивает новую голову (классификатор) под num_classes
        model = timm.create_model(
            timm_arch, 
            pretrained=pretrained, 
            num_classes=num_classes,
            drop_rate=drop_rate
        )
    except Exception as e:
        raise ValueError(f"Ошибка при создании модели '{timm_arch}' через timm: {e}")
        
    if freeze_backbone:
        # Замораживаем все веса в модели (бэкбоун)
        for param in model.parameters():
            param.requires_grad = False
            
        # timm предоставляет удобный метод reset_classifier. 
        # Вызов этого метода пересоздает голову, и новые веса автоматически получают requires_grad=True
        model.reset_classifier(num_classes)
        print(f"Бэкбоун заморожен. Обучаться будет только классификационная голова.")
    else:
        print("Вся модель (бэкбоун + голова) доступна для обучения.")
        
    return model

if __name__ == "__main__":
    # Локальный тест
    cfg = {
        'model': {
            'architecture': 'efficientnet-b0',
            'num_classes': 2,
            'pretrained': False,
            'freeze_backbone': True,
            'drop_rate': 0.3
        }
    }
    model = create_model(cfg)
    
    # Проверка, какие параметры обучаются
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Обучаемых параметров: {trainable_params}")
