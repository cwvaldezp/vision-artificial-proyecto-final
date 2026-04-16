import os
import torch
import matplotlib.pyplot as plt


def calculate_accuracy(outputs, labels):
    """
    Calcula el accuracy de un batch.

    Parámetros:
    - outputs: salida del modelo, forma [batch_size, num_classes]
    - labels: etiquetas reales, forma [batch_size]

    Proceso:
    - tomamos la clase con mayor puntaje usando argmax
    - comparamos contra las etiquetas reales
    - calculamos el porcentaje de aciertos

    Retorna:
    - accuracy del batch en formato decimal
      Ejemplo: 0.75 = 75%
    """
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Entrena el modelo durante una época completa.

    Parámetros:
    - model: red neuronal
    - dataloader: DataLoader de entrenamiento
    - criterion: función de pérdida (ej. CrossEntropyLoss)
    - optimizer: optimizador (ej. Adam)
    - device: cpu o cuda

    Retorna:
    - epoch_loss: pérdida promedio de la época
    - epoch_acc: accuracy promedio de la época
    """
    # Ponemos el modelo en modo entrenamiento
    # Esto es importante porque activa comportamientos como dropout o batchnorm en modo train
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    total_batches = 0

    for images, labels in dataloader:
        # Enviamos imágenes y etiquetas al dispositivo correspondiente
        images = images.to(device)
        labels = labels.to(device)

        # Ponemos en cero los gradientes acumulados de la iteración anterior
        optimizer.zero_grad()

        # Forward pass:
        # el modelo genera una predicción para cada imagen del batch
        outputs = model(images)

        # Calculamos la pérdida comparando la salida del modelo con las etiquetas reales
        loss = criterion(outputs, labels)

        # Backpropagation:
        # calculamos gradientes de la pérdida respecto a los pesos
        loss.backward()

        # Actualizamos los pesos del modelo
        optimizer.step()

        # Acumulamos métricas para luego sacar promedios de toda la época
        running_loss += loss.item()
        running_acc += calculate_accuracy(outputs, labels)
        total_batches += 1

    # Promedio de pérdida y accuracy de la época
    epoch_loss = running_loss / total_batches
    epoch_acc = running_acc / total_batches

    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Evalúa el modelo durante una época completa en validación o test.

    Diferencia clave respecto a train:
    - no se actualizan los pesos
    - se usa model.eval()
    - se usa torch.no_grad() para ahorrar memoria y acelerar evaluación

    Retorna:
    - epoch_loss: pérdida promedio
    - epoch_acc: accuracy promedio
    """
    # Ponemos el modelo en modo evaluación
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    total_batches = 0

    # Desactivamos cálculo de gradientes porque no vamos a entrenar
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            running_acc += calculate_accuracy(outputs, labels)
            total_batches += 1

    epoch_loss = running_loss / total_batches
    epoch_acc = running_acc / total_batches

    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10,
                save_path="outputs/models/best_model.pth"):
    """
    Función principal de entrenamiento.

    Qué hace:
    - entrena el modelo por varias épocas
    - valida en cada época
    - guarda el mejor modelo según validation accuracy
    - devuelve un historial con métricas

    Parámetros:
    - model: modelo a entrenar
    - train_loader: DataLoader de entrenamiento
    - val_loader: DataLoader de validación
    - criterion: función de pérdida
    - optimizer: optimizador
    - device: cpu o cuda
    - epochs: número de épocas
    - save_path: ruta donde guardar el mejor modelo

    Retorna:
    - history: diccionario con pérdidas y accuracies por época
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0

    # Creamos la carpeta del modelo si no existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # Guardamos métricas en el historial
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Si el accuracy de validación mejora, guardamos el modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Mejor modelo guardado en: {save_path}")

    return history


def plot_training_history(history, save_dir=None):
    """
    Grafica las métricas de entrenamiento y validación.

    Qué muestra:
    - pérdida de entrenamiento vs validación
    - accuracy de entrenamiento vs validación

    Parámetros:
    - history: diccionario retornado por train_model
    - save_dir: carpeta opcional para guardar las gráficas
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # -------- Gráfico de pérdida --------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Pérdida de entrenamiento y validación")
    plt.legend()
    plt.grid(True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "loss_curve.png"), bbox_inches="tight")

    plt.show()

    # -------- Gráfico de accuracy --------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title("Accuracy de entrenamiento y validación")
    plt.legend()
    plt.grid(True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), bbox_inches="tight")

    plt.show()


def load_model_weights(model, weights_path, device):
    """
    Carga pesos guardados previamente en un modelo.

    Parámetros:
    - model: instancia del modelo
    - weights_path: ruta del archivo .pth
    - device: cpu o cuda

    Retorna:
    - model con pesos cargados
    """
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_single_image(model, image_tensor, device):
    """
    Predice la clase de una sola imagen.

    Parámetros:
    - model: modelo entrenado
    - image_tensor: tensor de una imagen con forma [C, H, W]
    - device: cpu o cuda

    Retorna:
    - pred_class: índice de la clase predicha
    """
    model.eval()

    with torch.no_grad():
        # Agregamos dimensión batch: [C, H, W] -> [1, C, H, W]
        image_tensor = image_tensor.unsqueeze(0).to(device)

        outputs = model(image_tensor)
        pred_class = torch.argmax(outputs, dim=1).item()

    return pred_class


def show_predictions(model, dataset, device, index_to_province, num_images=5):
    """
    Muestra varias imágenes del dataset junto con su etiqueta real y predicción.

    Parámetros:
    - model: modelo entrenado
    - dataset: dataset cargado
    - device: cpu o cuda
    - index_to_province: diccionario {indice: nombre_provincia}
    - num_images: cantidad de imágenes a mostrar

    Nota:
    - esta función es útil para el entregable, porque el profe pide mostrar
      una imagen clasificada para demostrar que el modelo funciona.
    """
    model.eval()

    plt.figure(figsize=(15, 4 * num_images))

    for i in range(num_images):
        image, label = dataset[i]

        with torch.no_grad():
            pred = predict_single_image(model, image, device)

        # Convertimos tensor a formato visualizable para matplotlib
        img_np = image.permute(1, 2, 0).cpu().numpy()

        plt.subplot(num_images, 1, i + 1)
        plt.imshow(img_np)
        plt.axis("off")
        plt.title(
            f"Real: {index_to_province[label]} | Predicción: {index_to_province[pred]}"
        )

    plt.tight_layout()
    plt.show()