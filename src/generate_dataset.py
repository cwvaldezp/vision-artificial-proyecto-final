import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# =========================================================
# 1. Mapeo oficial de letras válidas -> provincias
# =========================================================
# Estas son las letras que sí representan provincias reales
# según el enunciado del proyecto final.
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

VALID_FIRST_LETTERS = list(LETTER_TO_PROVINCE.keys())


# =========================================================
# 2. Funciones auxiliares para generar texto de placa
# =========================================================
def random_letters(k: int) -> str:
    """
    Genera k letras aleatorias en mayúsculas.
    """
    return "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=k))


def random_numbers(k: int) -> str:
    """
    Genera k dígitos aleatorios.
    """
    return "".join(random.choices("0123456789", k=k))


def generate_plate_number():
    """
    Genera un número de placa con formato válido para este proyecto.

    Mejora respecto al generador base:
    - La primera letra SIEMPRE pertenece a una provincia real.
    - Las otras letras pueden variar libremente.

    Ejemplos:
    - PBC-1234
    - ADF-782
    """
    first_letter = random.choice(VALID_FIRST_LETTERS)

    formatos = [
        lambda: f"{first_letter}{random_letters(2)}-{random_numbers(3)}",
        lambda: f"{first_letter}{random_letters(2)}-{random_numbers(4)}",
    ]

    return random.choice(formatos)()


# =========================================================
# 3. Función para renderizar visualmente una placa
# =========================================================
def render_plate(
    text,
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    size=(404, 154),
    apply_augmentation=True
):
    """
    Crea una imagen sintética de una placa vehicular.

    Parámetros:
    - text: texto de la placa, por ejemplo "ABC-1234"
    - font_path: ruta de la tipografía
    - size: tamaño de la imagen
    - apply_augmentation: si True, aplica pequeñas variaciones visuales

    Retorna:
    - imagen PIL
    """
    # Colores base
    white = (255, 255, 255)
    black = (0, 0, 0)
    dark_blue = (0, 38, 84)
    light_gray = (235, 235, 235)

    # Crear lienzo base de la placa
    img = Image.new("RGB", size, color=white)
    draw = ImageDraw.Draw(img)

    width, height = size

    # Banda superior tipo placa Ecuador
    band_height = int(height * 0.22)
    draw.rectangle([0, 0, width, band_height], fill=light_gray)

    # Borde externo
    draw.rectangle([0, 0, width - 1, height - 1], outline=black, width=2)

    # Texto "ECUADOR" arriba
    try:
        country_font = ImageFont.truetype(font_path, 22)
    except:
        country_font = ImageFont.load_default()

    country_text = "ECUADOR"
    bbox_country = draw.textbbox((0, 0), country_text, font=country_font)
    country_w = bbox_country[2] - bbox_country[0]
    country_h = bbox_country[3] - bbox_country[1]

    draw.text(
        ((width - country_w) // 2, (band_height - country_h) // 2 - 2),
        country_text,
        fill=dark_blue,
        font=country_font
    )

    # Texto principal de la placa
    try:
        plate_font = ImageFont.truetype(font_path, 64)
    except:
        plate_font = ImageFont.load_default()

    bbox_plate = draw.textbbox((0, 0), text, font=plate_font)
    plate_w = bbox_plate[2] - bbox_plate[0]
    plate_h = bbox_plate[3] - bbox_plate[1]

    x_text = (width - plate_w) // 2
    y_text = band_height + ((height - band_height - plate_h) // 2) - 5

    draw.text((x_text, y_text), text, fill=black, font=plate_font)

    # =====================================================
    # 4. Mejoras / augmentations visuales ligeras
    # =====================================================
    # Esto ayuda a que el dataset no sea tan artificialmente perfecto.
    if apply_augmentation:
        # Pequeña rotación aleatoria
        angle = random.uniform(-3, 3)
        img = img.rotate(angle, expand=False, fillcolor=white)

        # Desenfoque suave ocasional
        if random.random() < 0.30:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

    return img


# =========================================================
# 5. Función principal para generar dataset
# =========================================================
def generate_dataset(
    output_dir="data/raw/synthethic_plates",
    num_images=20000,
    train_ratio=0.8,
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
):
    """
    Genera un dataset sintético de placas y lo divide en train/test.

    Parámetros:
    - output_dir: carpeta raíz de salida
    - num_images: cantidad total de imágenes a generar
    - train_ratio: proporción para entrenamiento
    - font_path: ruta de la fuente tipográfica
    """

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    # Crear carpetas si no existen
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    generated_plates = set()

    # Vamos a generar placas únicas para evitar duplicados
    while len(generated_plates) < num_images:
        plate = generate_plate_number()

        if plate in generated_plates:
            continue

        generated_plates.add(plate)

        img = render_plate(
            text=plate,
            font_path=font_path,
            apply_augmentation=True
        )

        # Decidir si va a train o test
        if random.random() < train_ratio:
            save_path = os.path.join(train_dir, f"{plate}.png")
        else:
            save_path = os.path.join(test_dir, f"{plate}.png")

        img.save(save_path)

    print(f"✅ Dataset generado correctamente en: {output_dir}")
    print(f"✅ Total imágenes: {num_images}")
    print(f"✅ Train dir: {train_dir}")
    print(f"✅ Test dir:  {test_dir}")


# =========================================================
# 6. Punto de entrada del script
# =========================================================
if __name__ == "__main__":
    generate_dataset(
        output_dir="data/raw/synthethic_plates",
        num_images=20000,
        train_ratio=0.8
    )