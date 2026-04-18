import os
import random
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance


# Mapeo de letras que representan a una provincia
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

# Fuentes disponibles con las que vamos a generar las placas
POSSIBLE_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]


# Funciones para la generacion de placas
# random.choices permite que las letras se repitan (por ejemplo, puede salir "AAA")
# .join une esos elementos en un solo bloque de texto (string). Ejemplo: "XRT".
def random_letters(k: int) -> str:
    return "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=k))


def random_numbers(k: int) -> str:
    return "".join(random.choices("0123456789", k=k))


def generate_plate_number():
    """
    Genera una placa donde la primera letra si esta mapeada en el LETTER_TO_PROVINCE.
    """
    first_letter = random.choice(VALID_FIRST_LETTERS)

    formats = [
        lambda: f"{first_letter}{random_letters(2)}-{random_numbers(3)}",
        lambda: f"{first_letter}{random_letters(2)}-{random_numbers(4)}",
    ]

    return random.choice(formats)()


def get_random_font(font_size: int):
    """
    Intenta cargar aleatoriamente una de las fuentes de POSSIBLE_FONTS.
    Si falla, usa la fuente por defecto.
    """
    fonts = POSSIBLE_FONTS.copy()
    random.shuffle(fonts)

    for font_path in fonts:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except:
                continue

    return ImageFont.load_default()


# Efectos visuales realistas
def add_sensor_noise(image, noise_std=6):
    """
    Agrega ruido suave tipo sensor de cámara.
    Mucho más realista que ruido RGB extremo.
    """
    img_np = np.array(image).astype(np.float32)
    
    # Generación del Ruido Gaussiano
    noise = np.random.normal(loc=0.0, scale=noise_std, size=img_np.shape)
    img_np = img_np + noise

    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def add_brightness_gradient(image):
    """
    Agrega un gradiente suave para simular iluminación desigual.
    """
    img_np = np.array(image).astype(np.float32)
    h, w, _ = img_np.shape

    # Gradiente horizontal o vertical
    if random.random() < 0.5:
        grad = np.linspace(random.uniform(0.85, 1.05), random.uniform(0.95, 1.15), w)
        grad = np.tile(grad, (h, 1))
    else:
        grad = np.linspace(random.uniform(0.85, 1.05), random.uniform(0.95, 1.15), h)
        grad = np.tile(grad[:, None], (1, w))

    grad = np.expand_dims(grad, axis=2)

    img_np = img_np * grad
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def apply_jpeg_compression(image, quality_range=(25, 75)):
    """
    Simula compresión típica de imágenes reales / WhatsApp / cámara.
    """
    quality = random.randint(*quality_range)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def add_shadow_overlay(image):
    """
    Agrega una sombra tenue sobre una parte de la placa.
    """
    img_np = np.array(image).astype(np.float32)
    h, w, _ = img_np.shape

    shadow = np.ones((h, w), dtype=np.float32)

    if random.random() < 0.5:
        # sombra lateral
        start = random.randint(0, w // 2)
        end = random.randint(start + 20, w)
        darkness = random.uniform(0.75, 0.95)
        shadow[:, start:end] *= darkness
    else:
        # sombra superior / inferior
        start = random.randint(0, h // 2)
        end = random.randint(start + 10, h)
        darkness = random.uniform(0.75, 0.95)
        shadow[start:end, :] *= darkness

    shadow = np.expand_dims(shadow, axis=2)
    img_np = img_np * shadow
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    return Image.fromarray(img_np)


def find_perspective_coeffs(src_pts, dst_pts):
    """
    Calcula coeficientes para transformación de perspectiva.
    """
    matrix = []
    for p1, p2 in zip(dst_pts, src_pts):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.array(matrix, dtype=np.float32)
    B = np.array(src_pts).reshape(8)

    coeffs = np.linalg.solve(A, B)
    return coeffs.tolist()


def apply_perspective_transform(image):
    """
    Aplica una perspectiva un poco más agresiva que la v2,
    pero aún controlada.
    """
    w, h = image.size

    mx = random.randint(6, 24)
    my = random.randint(4, 14)

    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (random.randint(0, mx), random.randint(0, my)),
        (w - random.randint(0, mx), random.randint(0, my)),
        (w - random.randint(0, mx), h - random.randint(0, my)),
        (random.randint(0, mx), h - random.randint(0, my)),
    ]

    coeffs = find_perspective_coeffs(src, dst)

    return image.transform(
        (w, h),
        Image.PERSPECTIVE,
        coeffs,
        resample=Image.BICUBIC,
        fillcolor=(235, 235, 235)
    )


# Render de placa
def render_plate(text, size=(404, 154), apply_augmentation=True):
    """
    Genera una placa sintética más parecida a una foto real recortada.
    """
    width, height = size

    # Fondo base menos perfecto
    base_white = random.randint(228, 250)
    plate_bg = (base_white, base_white, base_white)

    # Banda superior
    gray_value = random.randint(215, 238)
    band_color = (gray_value, gray_value, gray_value)

    # Borde
    border_dark = random.randint(20, 70)
    border_color = (border_dark, border_dark, border_dark)

    # Texto principal
    txt_dark = random.randint(0, 35)
    text_color = (txt_dark, txt_dark, txt_dark)

    # ECUADOR
    blue_color = (
        random.randint(0, 30),
        random.randint(60, 110),
        random.randint(110, 180)
    )

    img = Image.new("RGB", size, color=plate_bg)
    draw = ImageDraw.Draw(img)

    # Banda superior
    band_height = random.randint(int(height * 0.16), int(height * 0.23))
    draw.rectangle([0, 0, width, band_height], fill=band_color)

    # Borde con grosor variable
    border_width = random.randint(1, 3)
    draw.rectangle([0, 0, width - 1, height - 1], outline=border_color, width=border_width)

    # Tornillos opcionales
    if random.random() < 0.6:
        screw_y = random.randint(18, 30)
        for screw_x in [random.randint(20, 40), random.randint(width - 40, width - 20)]:
            r = random.randint(2, 4)
            screw_color = tuple([random.randint(70, 130)] * 3)
            draw.ellipse(
                [screw_x - r, screw_y - r, screw_x + r, screw_y + r],
                fill=screw_color
            )

    # Fuentes
    country_font = get_random_font(random.randint(16, 24))
    plate_font = get_random_font(random.randint(54, 74))

    # Texto ECUADOR
    country_text = "ECUADOR"
    bbox_country = draw.textbbox((0, 0), country_text, font=country_font)
    country_w = bbox_country[2] - bbox_country[0]
    country_h = bbox_country[3] - bbox_country[1]

    country_x = (width - country_w) // 2 + random.randint(-10, 10)
    country_y = max(2, (band_height - country_h) // 2 + random.randint(-2, 3))

    draw.text((country_x, country_y), country_text, fill=blue_color, font=country_font)

    # Texto principal
    bbox_plate = draw.textbbox((0, 0), text, font=plate_font)
    plate_w = bbox_plate[2] - bbox_plate[0]
    plate_h = bbox_plate[3] - bbox_plate[1]

    x_jitter = random.randint(-22, 22)
    y_jitter = random.randint(-10, 10)

    x_text = (width - plate_w) // 2 + x_jitter
    y_text = band_height + ((height - band_height - plate_h) // 2) - 4 + y_jitter

    # Dibujar texto 1 o 2 veces muy cerca para variar grosor aparente
    draw.text((x_text, y_text), text, fill=text_color, font=plate_font)
    if random.random() < 0.25:
        draw.text((x_text + 1, y_text), text, fill=text_color, font=plate_font)

    if apply_augmentation:
        # Perspectiva
        if random.random() < 0.45:
            img = apply_perspective_transform(img)

        # Rotación leve
        if random.random() < 0.65:
            angle = random.uniform(-5, 5)
            img = img.rotate(
                angle,
                expand=False,
                resample=Image.BICUBIC,
                fillcolor=plate_bg
            )

        # Brillo
        if random.random() < 0.50:
            img = ImageEnhance.Brightness(img).enhance(
                random.uniform(0.88, 1.12)
            )

        # Contraste
        if random.random() < 0.50:
            img = ImageEnhance.Contrast(img).enhance(
                random.uniform(0.90, 1.15)
            )

        # Gradiente de luz
        if random.random() < 0.25:
            img = add_brightness_gradient(img)

        # Sombra
        if random.random() < 0.15:
            img = add_shadow_overlay(img)

        # Blur leve a moderado
        if random.random() < 0.25:
            img = img.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(0.2, 1.0)
                )
            )

        # Ruido sensor
        if random.random() < 0.25:
            img = add_sensor_noise(
                img,
                noise_std=random.uniform(2.0, 6.0)
            )

        # Compresión JPEG
        if random.random() < 0.30:
            img = apply_jpeg_compression(
                img,
                quality_range=(35, 75)
            )

        # Crop parcial leve y reescalado
        if random.random() < 0.30:
            w, h = img.size
            left = random.randint(0, 10)
            top = random.randint(0, 5)
            right = random.randint(w - 10, w)
            bottom = random.randint(h - 5, h)

            cropped = img.crop((left, top, right, bottom))
            img = cropped.resize((w, h), Image.BICUBIC)

    return img


# Generación de dataset
def generate_dataset(
    output_dir="data/raw/synthetic_plates",
    num_images=30000,
    train_ratio=0.8
):
    """
    Genera dataset sintético y divide en train/test.
    """
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    generated_plates = set()
    train_count = 0
    test_count = 0

    while len(generated_plates) < num_images:
        plate = generate_plate_number()

        if plate in generated_plates:
            continue

        generated_plates.add(plate)

        img = render_plate(
            text=plate,
            apply_augmentation=True
        )

        if random.random() < train_ratio:
            save_path = os.path.join(train_dir, f"{plate}.png")
            train_count += 1
        else:
            save_path = os.path.join(test_dir, f"{plate}.png")
            test_count += 1

        img.save(save_path)

    print(f"Dataset generado correctamente en: {output_dir}")
    print(f"Total imágenes: {num_images}")
    print(f"Train: {train_count} ({(train_count/num_images)*100:.2f}%)")
    print(f"Test:  {test_count} ({(test_count/num_images)*100:.2f}%)")


# Main
if __name__ == "__main__":
    generate_dataset(
        output_dir="data/raw/synthetic_plates",
        num_images=30000,
        train_ratio=0.8
    )