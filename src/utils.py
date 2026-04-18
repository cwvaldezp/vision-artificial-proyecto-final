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
    - obtenemos la clase predicha usando argmax
    - comparamos contra la etiqueta real
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
    # Modo entrenamiento
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    total_batches = 0

    for images, labels in dataloader:
        # Enviar batch al dispositivo
        images = images.to(device)
        labels = labels.to(device)

        # Limpiar gradientes anteriores
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calcular pérdida
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Actualización de pesos
        optimizer.step()

        # Acumular métricas
        running_loss += loss.item()
        running_acc += calculate_accuracy(outputs, labels)
        total_batches += 1

    # Promedios de la época
    epoch_loss = running_loss / total_batches
    epoch_acc = running_acc / total_batches

    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Evalúa el modelo durante una época completa en validación o test.

    Diferencias respecto a entrenamiento:
    - no actualiza pesos
    - usa model.eval()
    - usa torch.no_grad()

    Retorna:
    - epoch_loss: pérdida promedio
    - epoch_acc: accuracy promedio
    """
    # Modo evaluación
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    total_batches = 0

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


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=10,
    save_path="outputs/models/best_model.pth",
    patience=10,
    writer=None
):
    """
    Función principal de entrenamiento.

    Qué hace:
    - entrena el modelo por varias épocas
    - valida en cada época
    - guarda el mejor modelo
    - aplica early stopping si deja de mejorar
    - guarda métricas en TensorBoard si se proporciona writer

    Parámetros:
    - model: modelo a entrenar
    - train_loader: DataLoader de entrenamiento
    - val_loader: DataLoader de validación
    - criterion: función de pérdida
    - optimizer: optimizador
    - device: cpu o cuda
    - epochs: número máximo de épocas
    - save_path: ruta donde guardar el mejor modelo
    - patience: número de épocas sin mejora antes de detener
    - writer: SummaryWriter de TensorBoard (opcional)

    Retorna:
    - history: diccionario con pérdidas y accuracies por época
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # Mejor accuracy observado hasta el momento
    best_val_acc = 0.0

    # Mejor loss observado hasta el momento
    # Sirve como criterio secundario cuando el accuracy empata
    best_val_loss = float("inf")

    # Contador para early stopping
    patience_counter = 0

    # Crear carpeta del modelo si no existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        
        # Entrenamiento
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validación
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # Guardar historial
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Guardar en TensorBoard si existe writer
        if writer is not None:
            writer.add_scalar("Loss/Train", train_loss, epoch + 1)
            writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
            writer.add_scalar("Accuracy/Train", train_acc, epoch + 1)
            writer.add_scalar("Accuracy/Validation", val_acc, epoch + 1)

        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Se considera mejora si:
        # 1. sube el validation accuracy
        # 2. o si el accuracy empata pero el validation loss baja
        improved = (
            (val_acc > best_val_acc) or
            (val_acc == best_val_acc and val_loss < best_val_loss)
        )

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Mejor modelo guardado en: {save_path}")

            # Reiniciar paciencia porque hubo mejora
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Sin mejora ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print("Early stopping activado")
                break

    # Cerrar TensorBoard writer
    if writer is not None:
        writer.close()

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
        # Agregar dimensión batch
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
    """
    model.eval()

    plt.figure(figsize=(15, 4 * num_images))

    for i in range(num_images):
        image, label = dataset[i]

        pred = predict_single_image(model, image, device)

        # Tensor -> numpy para visualizar con matplotlib
        img_np = image.permute(1, 2, 0).cpu().numpy()

        plt.subplot(num_images, 1, i + 1)
        plt.imshow(img_np)
        plt.axis("off")
        plt.title(
            f"Real: {index_to_province[label]} | Predicción: {index_to_province[pred]}"
        )

    plt.tight_layout()
    plt.show()