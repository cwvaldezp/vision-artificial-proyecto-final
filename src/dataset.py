import os
from glob import glob
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2


# =========================================================
# 1. Mapeo de letra inicial -> provincia
# =========================================================
# Según el enunciado del proyecto, la provincia de la placa
# se obtiene a partir de la PRIMERA letra del nombre del archivo.
#
# Ejemplo:
#   AAB-1269.png -> A -> Azuay
#
# Este diccionario nos permite convertir esa letra en el nombre
# real de la provincia.
LETTER_TO_PROVINCE = {
    "A": "Azuay",
    "B": "Bolivar",
    "C": "Carchi",
    "E": "Esmeraldas",
    "G": "Guayas",
    "H": "Chimborazo",
    "I": "Imbabura",
    "J": "Santo Domingo",
    "K": "Sucumbios",
    "L": "Loja",
    "M": "Manabi",
    "N": "Napo",
    "O": "El Oro",
    "P": "Pichincha",
    "Q": "Orellana",
    "R": "Los Rios",
    "S": "Pastaza",
    "T": "Tungurahua",
    "U": "Cañar",
    "V": "Morona Santiago",
    "W": "Galapagos",
    "X": "Cotopaxi",
    "Y": "Santa Elena",
    "Z": "Zamora Chinchipe",
}


# =========================================================
# 2. Conversión de letras a índices de clase
# =========================================================
# PyTorch trabaja mejor con clases numéricas:
#   A -> 0
#   B -> 1
#   ...
#
# Por eso creamos diccionarios auxiliares para convertir:
# - letra -> índice
# - índice -> letra
# - índice -> provincia
CLASSES = sorted(LETTER_TO_PROVINCE.keys())

LETTER_TO_INDEX = {
    letter: idx for idx, letter in enumerate(CLASSES)
}

INDEX_TO_LETTER = {
    idx: letter for letter, idx in LETTER_TO_INDEX.items()
}

INDEX_TO_PROVINCE = {
    idx: LETTER_TO_PROVINCE[letter] for letter, idx in LETTER_TO_INDEX.items()
}


# =========================================================
# 3. Transformaciones para entrenamiento
# =========================================================
# Estas transformaciones se aplican SOLO durante entrenamiento.
#
# Objetivo:
# - dar un poco de variación visual,
# - mejorar generalización,
# - pero sin destruir demasiado la placa.
#
# En la versión anterior las augmentations eran más agresivas.
# Aquí las suavizamos para que sigan siendo realistas.
train_transforms = v2.Compose([
    # Redimensionar todas las imágenes al mismo tamaño
    # para que entren uniformemente al modelo.
    v2.Resize((128, 256)),

    # Aplicar cambios leves de iluminación y contraste.
    # Esto ayuda a que el modelo no dependa de un único tono.
    v2.RandomApply([
        v2.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.05
        )
    ], p=0.5),

    # Rotación muy pequeña.
    # Esto simula una placa ligeramente inclinada,
    # pero evita convertirla en algo irreal.
    v2.RandomRotation(degrees=3),

    # Traslación y escala muy suaves.
    # Sirve para que el modelo no memorice una posición exacta.
    v2.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.97, 1.03)
    ),

    # Blur suave ocasional.
    # Simula una imagen no totalmente nítida.
    v2.RandomApply([
        v2.GaussianBlur(kernel_size=3)
    ], p=0.2),

    # Convierte la imagen a formato tensor-image moderno de torchvision
    v2.ToImage(),

    # Convierte a float32 y escala automáticamente a rango [0, 1]
    v2.ToDtype(torch.float32, scale=True)
])


# =========================================================
# 4. Transformaciones para validación y test
# =========================================================
# Aquí NO usamos augmentations aleatorias.
# Solo queremos medir el desempeño real del modelo sobre datos
# de validación/test sin alterar la imagen.
eval_transforms = v2.Compose([
    # Redimensionar al mismo tamaño usado en entrenamiento
    v2.Resize((128, 256)),

    # Convertir a tensor-image
    v2.ToImage(),

    # Convertir a float32 y escalar a [0, 1]
    v2.ToDtype(torch.float32, scale=True)
])


# =========================================================
# 5. Dataset personalizado
# =========================================================
class PlateProvinceDataset(Dataset):
    """
    Dataset para clasificar una placa vehicular en una de las 24 provincias.

    La clase se obtiene a partir de la primera letra del nombre del archivo.

    Ejemplo:
        AAB-1269.png -> 'A' -> Azuay
        PDH-2313.jpeg -> 'P' -> Pichincha
    """

    def __init__(
        self,
        image_dir: str,
        transform: Optional[v2.Compose] = None,
        exclude_invalid: bool = True
    ):
        """
        Parámetros:
        - image_dir: carpeta donde están las imágenes
        - transform: transformaciones a aplicar
        - exclude_invalid: si True, ignora archivos cuya primera letra
          no pertenezca a una provincia válida
        """
        # Guardar parámetros de entrada
        self.image_dir = image_dir
        self.transform = transform if transform is not None else eval_transforms
        self.exclude_invalid = exclude_invalid

        # Listas internas para rutas de imágenes y etiquetas numéricas
        self.image_paths = []
        self.labels = []

        # Validar que la carpeta exista para evitar errores silenciosos
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"No existe la carpeta del dataset: {image_dir}")

        # Patrones permitidos de archivo.
        # Así aceptamos png, jpg y jpeg, tanto en minúsculas como mayúsculas.
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]

        # Lista temporal donde iremos acumulando todos los archivos encontrados
        all_files = []

        # Buscar imágenes que cumplan cada patrón
        for pattern in patterns:
            all_files.extend(glob(os.path.join(image_dir, pattern)))

        # Ordenar para mantener consistencia entre ejecuciones
        all_files = sorted(all_files)

        # Recorrer todos los archivos encontrados
        for image_path in all_files:
            # Obtener solo el nombre del archivo
            filename = os.path.basename(image_path)

            # La clase se determina con la primera letra del nombre
            first_letter = filename[0].upper()

            # Si la letra no es válida, se ignora si exclude_invalid=True
            if first_letter not in LETTER_TO_INDEX:
                if self.exclude_invalid:
                    continue
                else:
                    # Aquí en el futuro se podría crear una clase especial
                    # como "NO_APLICA", si el proyecto lo necesitara.
                    continue

            # Guardar ruta y etiqueta numérica correspondiente
            self.image_paths.append(image_path)
            self.labels.append(LETTER_TO_INDEX[first_letter])

    def __len__(self):
        """
        Devuelve cuántas imágenes tiene el dataset.
        Esto permite que len(dataset) funcione correctamente.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Devuelve una muestra del dataset:

        Retorna:
        - image: imagen transformada
        - label: etiqueta numérica de la provincia
        """
        # Ruta de la imagen solicitada
        image_path = self.image_paths[idx]

        # Abrir imagen y asegurar formato RGB
        image = Image.open(image_path).convert("RGB")

        # Aplicar transformaciones si existen
        if self.transform:
            image = self.transform(image)

        # Obtener etiqueta numérica ya almacenada
        label = self.labels[idx]

        return image, label

    def get_class_name(self, idx: int) -> str:
        """
        Devuelve el nombre de la provincia a partir del índice de clase.

        Ejemplo:
        0 -> Azuay
        """
        return INDEX_TO_PROVINCE[idx]

    def get_letter(self, idx: int) -> str:
        """
        Devuelve la letra asociada al índice de clase.

        Ejemplo:
        0 -> A
        """
        return INDEX_TO_LETTER[idx]