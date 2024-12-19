import cv2
import os
import mediapipe as mp

# Путь к папке с исходными изображениями
source_folder = "uploaded_images"  # Укажите папку с загруженными изображениями
destination_folder = "dataset"  # Папка для сохранения обработанных данных

# Убедитесь, что папка dataset существует
os.makedirs(destination_folder, exist_ok=True)

# Инициализация MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Пройдемся по всем папкам в source_folder
for user_folder_name in os.listdir(source_folder):
    user_folder_path = os.path.join(source_folder, user_folder_name)

    # Проверяем, является ли это папкой
    if os.path.isdir(user_folder_path):
        print(f"\n[INFO] Обработка изображений для пользователя: {user_folder_name}")

        # Создаем папку для этого пользователя в destination_folder
        user_dataset_folder = os.path.join(destination_folder, user_folder_name)
        os.makedirs(user_dataset_folder, exist_ok=True)

        count = 0

        # Обработка каждого изображения из папки пользователя
        for file_name in os.listdir(user_folder_path):
            file_path = os.path.join(user_folder_path, file_name)

            # Проверяем, является ли файл изображением
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                print(f"[WARNING] Пропущен файл: {file_name} (не изображение)")
                continue

            # Чтение изображения
            img = cv2.imread(file_path)
            if img is None:
                print(f"[WARNING] Невозможно загрузить файл: {file_name}")
                continue

            # Конвертация в RGB и обработка с MediaPipe
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = img.shape
                    bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                           int(bboxC.width * w), int(bboxC.height * h)
                    x, y, w, h = bbox

                    # Убедитесь, что координаты в пределах изображения
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, img.shape[1] - x)
                    h = min(h, img.shape[0] - y)

                    # Извлечение лица
                    face_img = img[y:y + h, x:x + w]
                    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

                    # Сохранение изображения в датасет
                    count += 1
                    save_path = os.path.join(user_dataset_folder, f"{user_folder_name}_{count}.jpg")
                    cv2.imwrite(save_path, gray_face)
                    print(f"[INFO] Сохранено изображение: {save_path}")

            else:
                print(f"[WARNING] Лицо не обнаружено на изображении: {file_name}")

            if count >= 30:  # Ограничение количества изображений для одного пользователя
                break

        print(f"[INFO] Обработка для пользователя {user_folder_name} завершена. Создано {count} изображений.")

print("\n[INFO] Обработка завершена.")
