# models/efficientnet_timm.py
import timm
import torch.nn as nn


def create_efficientnet(model_name: str = "efficientnet_b0", pretrained: bool = True, num_classes: int = 2):
    """Crea un modelo EfficientNet usando timm y adapta la capa final para num_classes."""
    model = timm.create_model(model_name, pretrained=pretrained)
    # Detectar nombre de la capa classifier/conv_head según la arquitectura
    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'head'):
        if hasattr(model.head, 'fc'):
            in_features = model.head.fc.in_features
            model.head.fc = nn.Linear(in_features, num_classes)
        else:
        # fallback: try to find last Linear
            for name, m in reversed(list(model.named_modules())):
                if isinstance(m, nn.Linear):
                    in_features = m.in_features
                    setattr(model, name, nn.Linear(in_features, num_classes))
                    break
                else:
                    # último recurso: reemplazar global pooling + classifier if timm variant
                    try:
                        model.reset_classifier(num_classes)
                    except Exception:
                        raise RuntimeError("No se pudo adaptar la capa final del modelo; revisá la arquitectura")


    return model