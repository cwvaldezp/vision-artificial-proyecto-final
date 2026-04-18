# 🚗 Proyecto Final - Clasificación de Provincias a partir de Placas Vehiculares

**Universidad San Francisco de Quito**  
**Maestría en Inteligencia Artificial**  
**Materia:** Visión Artificial  

---

## 🎯 Objetivo

Desarrollar un modelo de **clasificación de imágenes** capaz de predecir la **provincia del Ecuador** a partir de una imagen de una placa vehicular.

La clasificación se basa en la **primera letra de la placa**, la cual representa la provincia.

---

## 🧠 Enfoque del Problema

Este proyecto se plantea como un problema de:

> 📌 **Clasificación multiclase (24 clases)**

Cada clase corresponde a una provincia del Ecuador.

Ejemplo:

| Placa        | Letra | Provincia   |
|-------------|------|------------|
| AAB-1269     | A    | Azuay      |
| PBC-7823     | P    | Pichincha  |
| GDF-123      | G    | Guayas     |

---

## 🗺️ Clases del Modelo

Las 24 clases corresponden a las siguientes provincias:

Azuay, Bolívar, Carchi, Chimborazo, Cotopaxi, El Oro, Esmeraldas,  
Galápagos, Guayas, Imbabura, Loja, Los Ríos, Manabí, Morona Santiago,  
Napo, Orellana, Pastaza, Pichincha, Santa Elena, Santo Domingo,  
Sucumbíos, Tungurahua, Zamora Chinchipe.

---

## 🏗️ Estructura del Proyecto
vision-artificial-proyecto-final/
├── data/
│ └── raw/
│ └── synthetic_plates/
│ ├── train/
│ └── test/
├── notebooks/
│ └── training.ipynb
├── src/
│ ├── dataset.py
│ ├── model.py
│ ├── utils.py
│ └── generate_dataset.py
├── outputs/
│ ├── models/
│ └── figures/
├── requirements.txt
└── README.md


---

## 📊 Dataset

Se utilizó un dataset **sintético de placas vehiculares**, generado a partir de un script basado en el visto en clases (`generate_dataset.py`).

### 🔧 Características del dataset:

- Generación automática de placas tipo: `ABC-1234`
- Primera letra válida según provincia del Ecuador
- Separación automática en:
  - **Train (80%)**
  - **Test (20%)**
- Inclusión de variaciones:
  - rotación leve
  - desenfoque (blur)

---

## 🤖 Modelo

Se utilizó un modelo de **Deep Learning basado en Transfer Learning**:

### 🔹 Arquitectura:
- **ResNet18 preentrenada en ImageNet**

### 🔹 Adaptación:
- Se reemplazó la última capa fully connected (`fc`)
- Se ajustó a **24 clases**

### 🔹 Ventajas:
- Aprovecha características visuales aprendidas previamente
- Reduce el tiempo de entrenamiento
- Mejora la generalización

---

## ⚙️ Entrenamiento

El entrenamiento se realizó en el notebook:
notebooks/training.ipynb


### 🔹 Incluye:
- Carga de dataset
- DataLoaders
- Entrenamiento por épocas
- Validación
- Cálculo de métricas:
  - Loss
  - Accuracy
- Visualización de resultados
- Ejemplos de predicción

---

## 📈 Resultados

Se analizaron:

- Curvas de pérdida (train vs validation)
- Curvas de accuracy (train vs validation)
- Ejemplos de clasificación visual

Esto permitió evaluar el desempeño del modelo y su capacidad de generalización.

---

## 💾 Modelo Entrenado

El modelo final se guarda en:
outputs/models/best_model.pth


Este archivo contiene los pesos entrenados del modelo.

---

## ▶️ Cómo ejecutar el proyecto

### 1. Clonar o ubicarse en el proyecto

```bash
cd vision-artificial-proyecto-final

2. (Opcional) Generar dataset
python src/generate_dataset.py

3. Ejecutar el entrenamiento
notebooks/training.ipynb

🧪 Ejemplo de Predicción

El modelo recibe una imagen de una placa:
PBC-7823.png

Salida esperada:
Predicción: Pichincha

🧠 Conclusiones
El uso de Transfer Learning permitió obtener buenos resultados rápidamente.
La generación de un dataset sintético facilitó el entrenamiento del modelo.
El modelo es capaz de identificar correctamente la provincia basándose en la primera letra de la placa.
🚀 Posibles Mejoras
Uso de modelos más avanzados (EfficientNet, MobileNet)
Aumentar diversidad del dataset
Incorporar ruido realista (iluminación, fondos)
Evaluación con imágenes reales
👨‍💻 Autor

Carlos Valdez
Maestría en Inteligencia Artificial - USFQ