import os
from glob import glob
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2


# Mapeo letra -> provincia, según el enunciado del proyecto
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

# Índices de clase para PyTorch
CLASSES = sorted(LETTER_TO_PROVINCE.keys())
LETTER_TO_INDEX = {letter: idx for idx, letter in enumerate(CLASSES)}
INDEX_TO_LETTER = {idx: letter for letter, idx in LETTER_TO_INDEX.items()}
INDEX_TO_PROVINCE = {idx: LETTER_TO_PROVINCE[letter] for letter, idx in LETTER_TO_INDEX.items()}


# Transformaciones por defecto para entrenamiento
train_transforms = v2.Compose([
    v2.Resize((128, 256)),
    v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)], p=0.7),
    v2.RandomRotation(degrees=5),
    v2.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

# Transformaciones para validación / test
eval_transforms = v2.Compose([
    v2.Resize((128, 256)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])


class PlateProvinceDataset(Dataset):
    """
    Dataset para clasificar una placa vehicular en una de las 24 provincias,
    usando la primera letra del nombre del archivo.
    Ejemplo:
        AAB-1269.png -> 'A' -> Azuay
    """

    def __init__(
        self,
        image_dir: str,
        transform: Optional[v2.Compose] = None,
        exclude_invalid: bool = True
    ):
        self.image_dir = image_dir
        self.transform = transform if transform is not None else eval_transforms
        self.exclude_invalid = exclude_invalid

        self.image_paths = []
        self.labels = []

        # Buscar imágenes .png, .jpg y .jpeg
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
        all_files = []
        for pattern in patterns:
            all_files.extend(glob(os.path.join(image_dir, pattern)))

        all_files = sorted(all_files)

        for image_path in all_files:
            filename = os.path.basename(image_path)
            first_letter = filename[0].upper()

            if first_letter not in LETTER_TO_INDEX:
                if self.exclude_invalid:
                    continue
                else:
                    # Si luego quieres soportar NO_APLICA, aquí sería el punto para agregarlo
                    continue

            self.image_paths.append(image_path)
            self.labels.append(LETTER_TO_INDEX[first_letter])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

    def get_class_name(self, idx: int) -> str:
        """Devuelve el nombre de la provincia para un índice de clase."""
        return INDEX_TO_PROVINCE[idx]

    def get_letter(self, idx: int) -> str:
        """Devuelve la letra asociada a un índice de clase."""
        return INDEX_TO_LETTER[idx]