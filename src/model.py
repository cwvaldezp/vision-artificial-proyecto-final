import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class PlateProvinceClassifier(nn.Module):
    """
    Modelo de clasificación de imágenes para placas vehiculares de Ecuador
    - Recibir una imagen de una placa
    - Predecir a qué provincia pertenece
    - La provincia se determina por la primera letra de la placa
    """

    def __init__(self, num_classes=24, use_pretrained=True, freeze_backbone=False):
        """
        Parámetros:
        - num_classes: número de clases de salida. En este proyecto son 24 provincias.
        - use_pretrained: si True, carga ResNet18 con pesos preentrenados en ImageNet.
        - freeze_backbone: si True, congela las capas convolucionales para entrenar solo la capa final.
        """
        super().__init__()

        # Si queremos usar pesos preentrenados, cargamos los pesos recomendados por torchvision.
        # Esto ayuda porque el modelo ya aprendió patrones visuales generales:
        # bordes, formas, texturas, contrastes, etc.
        weights = ResNet18_Weights.DEFAULT if use_pretrained else None

        # Cargamos la arquitectura base ResNet18
        self.backbone = resnet18(weights=weights)

        # fc = fully connected layer = última capa del modelo
        # Esa capa originalmente fue entrenada para clasificar 1000 clases de ImageNet
        in_features = self.backbone.fc.in_features

        # Reemplazamos la última capa por una nueva capa lineal
        # que ahora tendrá salida para nuestras 24 clases
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # No se uso porque tenemos un dataset suficientemente grande 
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

            # si congelamos todo, debemos volver a habilitar la capa final
            # para que sí aprenda nuestras clases
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        Forward pass:
        - x: batch de imágenes con forma [batch_size, 3, alto, ancho]
        - retorna: logits de forma [batch_size, num_classes]

        Los logits son valores sin softmax.
        CrossEntropyLoss ya aplica internamente la parte necesaria,
        por eso aquí NO usamos softmax.
        """
        return self.backbone(x)


def build_model(num_classes=24, use_pretrained=True, freeze_backbone=False):
    """
    Función auxiliar para construir el modelo.
    Sirve para mantener el notebook más limpio y ordenado.
    """
    model = PlateProvinceClassifier(
        num_classes=num_classes,
        use_pretrained=use_pretrained,
        freeze_backbone=freeze_backbone
    )
    return model


def get_device():
    """
    Detecta si hay GPU disponible.
    - Si hay CUDA, usamos GPU
    - Si no, usamos CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")