import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import os
import pickle

# Путь к изображениям
dataset_path = "./dataset"

# Сбор данных для обучения
def load_images_and_labels(dataset_path):
    images = []
    labels = []

    # Проходим по всем папкам внутри dataset
    for label_name in os.listdir(dataset_path):
        label_folder_path = os.path.join(dataset_path, label_name)

        # Проверяем, является ли это папкой
        if os.path.isdir(label_folder_path):
            print(f"[INFO] Обработка папки: {label_name}")
            for image_name in os.listdir(label_folder_path):
                image_path = os.path.join(label_folder_path, image_name)

                # Проверяем, является ли файл изображением
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    print(f"[WARNING] Пропущен файл: {image_name} (не изображение)")
                    continue

                # Чтение изображения
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"[WARNING] Невозможно загрузить файл: {image_path}")
                    continue

                # Приводим изображение к размеру 100x100
                img_resized = cv2.resize(img, (100, 100))

                # Добавляем изображение и метку
                images.append(img_resized.flatten())  # Преобразуем изображение в одномерный массив
                labels.append(label_name)  # Используем имя папки как метку

    return np.array(images), np.array(labels)

print("[INFO] Загрузка изображений...")
X, y = load_images_and_labels(dataset_path)

# [ДОБАВЛЕНО ДЛЯ ОТЛАДКИ]
print(f"[INFO] Количество изображений: {len(y)}")
print(f"[INFO] Уникальные метки: {set(y)}")

# Проверяем, есть ли больше одного класса
unique_classes = set(y)
if len(unique_classes) <= 1:
    if len(unique_classes) == 0:
        raise ValueError("В папке 'dataset' нет подходящих изображений. Убедитесь, что структура папки корректна.")
    else:
        raise ValueError(
            "В папке 'dataset' предоставлены данные только для одного класса. Добавьте данные для второго класса.")

# Кодировка меток
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Тренировка SVM
print("[INFO] Обучение модели...")
model = SVC(kernel="linear", probability=True)
model.fit(X, y_encoded)

# Сохранение модели
os.makedirs("trainer", exist_ok=True)
with open("trainer/face_recognition_model.pkl", "wb") as f:
    pickle.dump((model, label_encoder), f)

print("[INFO] Обучение завершено. Модель сохранена.")
