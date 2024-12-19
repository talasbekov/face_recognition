import cv2
import mediapipe as mp
import os
import time

# Создаём папку для сохранения лиц, если её ещё нет
output_folder = "kpp2"
os.makedirs(output_folder, exist_ok=True)

# Инициализация Mediapipe для распознавания лиц
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Подключение к веб-камере
cap = cv2.VideoCapture(0)  # Используется стандартная веб-камера (индекс 0)

# Устанавливаем более длительный таймаут для захвата видео
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Уменьшаем размер буфера

# Счётчик для именования сохранённых изображений
image_counter = 0
timeout = 20
start_time = time.time()

# Функция для попытки переподключения к видео потоку
def reconnect_to_camera():
    print("Попытка переподключения к камере...")
    cap.release()
    time.sleep(5)
    cap.open(0)  # Повторное подключение к веб-камере
    return cap.isOpened()

# Инициализация Mediapipe
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            if time.time() - start_time > timeout:
                print("Не удалось получить кадр с камеры. Попытка переподключения...")
                if not reconnect_to_camera():
                    print("Не удалось восстановить подключение к камере.")
                    break
                start_time = time.time()
            continue

        start_time = time.time()

        # Конвертируем изображение в RGB для обработки Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Обратно в BGR для отображения
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Рисуем рамки вокруг лиц и сохраняем каждое лицо
        if results.detections:
            for detection in results.detections:
                # Получаем координаты рамки
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Обрезаем область лица
                face = frame[y:y + h, x:x + w]

                # Сохраняем лицо в папку
                if face.size > 0:  # Убедимся, что лицо не пустое
                    face_path = os.path.join(output_folder, f"face_{image_counter}.jpg")
                    cv2.imwrite(face_path, face)
                    print(f"Лицо сохранено: {face_path}")
                    image_counter += 1

                # Рисуем рамку вокруг лица на основном изображении
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Отображаем кадр с наложенными рамками
        cv2.imshow('Распознавание лиц', image)

        # Нажмите 'q', чтобы выйти из цикла
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
