import cv2
import mediapipe as mp
import pickle


# Загрузка обученной модели
with open("trainer/face_recognition_model.pkl", "rb") as f:
    model, label_encoder = pickle.load(f)

# Инициализация MediaPipe для обнаружения лиц
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Запуск видеонаблюдения с веб-камеры
cam = cv2.VideoCapture(0)
cam.set(3, 1280)  # ширина кадра
cam.set(4, 720)  # высота кадра

# Настройка окна отображения
cv2.namedWindow("Видеонаблюдение", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Видеонаблюдение", 1280, 720)  # Размер окна

print("\n[INFO] Видеонаблюдение запущено. Нажмите 'ESC' для выхода.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("[ERROR] Камера недоступна.")
        break

    # Преобразование кадра в RGB для работы с MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    # Если лица обнаружены
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)
            x, y, w, h = bbox

            # Убедимся, что координаты находятся в пределах кадра
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            # Извлечение изображения лица
            face_img = frame[y:y + h, x:x + w]
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray_face, (100, 100))  # Размер 100x100 как при обучении
            face_vector = gray_resized.flatten().reshape(1, -1)  # Преобразуем в вектор

            # Распознавание лица
            try:
                prediction = model.predict(face_vector)
                confidence = model.predict_proba(face_vector).max() * 100
                name = label_encoder.inverse_transform(prediction)[0]
            except Exception as e:
                print(f"[ERROR] Ошибка распознавания: {e}")
                name = "Неизвестно"
                confidence = 0

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}: {confidence:.2f}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Показ видео с аннотацией
    cv2.imshow("Видеонаблюдение", frame)

    # Нажмите ESC для выхода
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("\n[INFO] Завершение работы.")
cam.release()
cv2.destroyAllWindows()
