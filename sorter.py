import os
import shutil
import argparse
import numpy as np
import tensorflow as tf
import cv2
import pathlib
import tempfile # ИЗМЕНЕНИЕ: импортируем модуль для безопасных временных папок

# --- Глобальные константы ---
# ... (остаются без изменений)
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
VIDEO_FRAMES_TO_EXTRACT = 10
VIDEO_FRAME_INTERVAL = 5
SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
SUPPORTED_VIDEO_FORMATS = ('.mp4', '.avi', '.mov', '.mkv')
CLASS_GOOD = 'good'
CLASS_TRASH = 'trash'


# ИЗМЕНЕНИЕ: Функция теперь принимает путь к временной папке
def prepare_training_data(good_examples_dir, trash_examples_dir, temp_dir_path, debug=False):
    """
    Готовит данные для обучения в предоставленной временной папке.
    """
    temp_dir = pathlib.Path(temp_dir_path)
    good_dest = temp_dir / CLASS_GOOD
    trash_dest = temp_dir / CLASS_TRASH
    
    os.makedirs(good_dest)
    os.makedirs(trash_dest)

    def copy_files(src_dir, dest_dir):
        for item in os.listdir(src_dir):
            s = os.path.join(src_dir, item)
            d = os.path.join(dest_dir, item)
            if os.path.isfile(s):
                shutil.copy2(s, d)

    if debug:
        print(f"DEBUG: Копирование хороших примеров из '{good_examples_dir}' в '{good_dest}'")
    copy_files(good_examples_dir, good_dest)
    
    if debug:
        print(f"DEBUG: Копирование мусорных примеров из '{trash_examples_dir}' в '{trash_dest}'")
    copy_files(trash_examples_dir, trash_dest)

    good_count = len(list(good_dest.glob('*.*')))
    trash_count = len(list(trash_dest.glob('*.*')))
    
    if debug:
        print(f"DEBUG: Подготовлено {good_count} хороших и {trash_count} мусорных примеров.")

    if good_count == 0 or trash_count == 0:
        print("ОШИБКА: Папки с примерами не содержат файлов. Невозможно обучить модель.")
        return False
        
    return True

# Функции train_or_load_model, train_new_model, extract_frames_from_video, classify_media
# остаются без изменений. Я их скрою для краткости, но они должны быть в вашем файле.
def train_or_load_model(training_data_dir, model_path="media_sorter_model.keras", debug=False):
    if os.path.exists(model_path):
        if debug:
            print(f"DEBUG: Загрузка существующей модели из '{model_path}'")
        try:
            model = tf.keras.models.load_model(model_path)
            if model is not None and len(model.layers) > 0:
                print("Валидация: Модель успешно загружена.")
                return model
            else:
                raise ValueError("Loaded model is invalid.")
        except Exception as e:
            print(f"ОШИБКА: Не удалось загрузить модель ({e}). Попробуем обучить заново.")
            return train_new_model(training_data_dir, model_path, debug)
    return train_new_model(training_data_dir, model_path, debug)

def train_new_model(data_dir, model_path, debug=False):
    print("Обучение новой модели. Это может занять некоторое время...")
    data_dir = pathlib.Path(data_dir)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE)
    class_names = train_ds.class_names
    print(f"Найдены классы: {class_names}")
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    num_classes = len(class_names)
    model = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, name="outputs")])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    epochs = 15
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2 if debug else 0)
    print("Обучение завершено.")
    model.save(model_path)
    if debug: print(f"DEBUG: Модель сохранена в '{model_path}'")
    return model

def extract_frames_from_video(video_path, num_frames, interval, debug=False):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if debug: print(f"DEBUG: Ошибка открытия видео файла: {video_path}")
        return frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        if debug: print(f"DEBUG: Не удалось получить FPS для видео: {video_path}. Используем значение по умолчанию 30.")
        fps = 30
    frame_interval = int(fps * interval)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    if interval > 0:
        frame_indices = range(0, total_frames, frame_interval)
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(rgb_frame, (IMG_WIDTH, IMG_HEIGHT))
            frames.append(resized_frame)
    cap.release()
    return frames

def classify_media(file_path, model, class_names, debug=False):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension in SUPPORTED_IMAGE_FORMATS:
        img = tf.keras.utils.load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        if debug: print(f"DEBUG: Изображение '{os.path.basename(file_path)}' -> Класс: {predicted_class}, Уверенность: {100 * np.max(score):.2f}%")
        return predicted_class
    elif file_extension in SUPPORTED_VIDEO_FORMATS:
        if debug: print(f"DEBUG: Обработка видео '{os.path.basename(file_path)}'...")
        frames = extract_frames_from_video(file_path, VIDEO_FRAMES_TO_EXTRACT, VIDEO_FRAME_INTERVAL, debug)
        if not frames:
            if debug: print(f"DEBUG: Не удалось извлечь кадры из '{os.path.basename(file_path)}'. Пропускаем.")
            return None
        frames_array = np.array(frames)
        predictions = model.predict(frames_array, verbose=0)
        votes = {class_name: 0 for class_name in class_names}
        for pred in predictions:
            votes[class_names[np.argmax(tf.nn.softmax(pred))]] += 1
        final_class = max(votes, key=votes.get)
        if debug: print(f"DEBUG: Видео '{os.path.basename(file_path)}' -> Голоса: {votes}, Итоговый класс: {final_class}")
        return final_class
    return None


def main():
    parser = argparse.ArgumentParser(description="Сортировка фото и видео с помощью TensorFlow.")
    parser.add_argument("--good_dir", required=True, help="Папка с примерами хороших медиа.")
    parser.add_argument("--trash_dir", required=True, help="Папка с примерами мусорных медиа.")
    parser.add_argument("--source_dir", required=True, help="Папка с файлами для сортировки.")
    parser.add_argument("--debug", action="store_true", help="Включить подробный вывод (DEBUG флаг).")
    args = parser.parse_args()

    # ИЗМЕНЕНИЕ: Используем `with` для автоматического создания и удаления временной папки.
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.debug:
            print(f"DEBUG: Создана безопасная временная папка: {temp_dir}")

        # --- 1. Подготовка данных ---
        print("--- Этап 1: Подготовка данных для обучения ---")
        if not prepare_training_data(args.good_dir, args.trash_dir, temp_dir, args.debug):
            return # Прерываем выполнение, если данные не подготовлены
        print("Валидация: Данные для обучения успешно подготовлены.\n")
        
        # --- 2. Обучение или загрузка модели ---
        print("--- Этап 2: Обучение или загрузка модели ---")
        class_names = sorted([CLASS_GOOD, CLASS_TRASH]) 
        model = train_or_load_model(temp_dir, debug=args.debug)
        if model is None:
            print("ОШИБКА: Не удалось создать или загрузить модель. Прерывание работы.")
            return
        print("Валидация: Модель готова к работе.\n")

        # --- 3. Сортировка и перемещение ---
        print(f"--- Этап 3: Сортировка файлов из '{args.source_dir}' ---")
        output_good_dir = os.path.join(args.source_dir, "sorted_good")
        output_trash_dir = os.path.join(args.source_dir, "sorted_trash")
        os.makedirs(output_good_dir, exist_ok=True)
        os.makedirs(output_trash_dir, exist_ok=True)
        
        moved_count, skipped_count = 0, 0
        all_files = [f for f in os.listdir(args.source_dir) if os.path.isfile(os.path.join(args.source_dir, f))]
        total_files = len(all_files)
        
        for i, filename in enumerate(all_files):
            file_path = os.path.join(args.source_dir, filename)
            print(f"Обработка [{i+1}/{total_files}]: {filename}")
            
            predicted_class = classify_media(file_path, model, class_names, args.debug)
            if predicted_class == CLASS_GOOD:
                shutil.move(file_path, os.path.join(output_good_dir, filename))
                moved_count += 1
            elif predicted_class == CLASS_TRASH:
                shutil.move(file_path, os.path.join(output_trash_dir, filename))
                moved_count += 1
            else:
                skipped_count += 1

        print("\nВалидация: Сортировка и перемещение завершены.")
        
        # --- 4. Отчет ---
        print("\n--- Итоговый отчет ---")
        print(f"Всего обработано файлов: {total_files}")
        print(f"Перемещено: {moved_count}")
        print(f"  - В папку 'sorted_good': {len(os.listdir(output_good_dir))}")
        print(f"  - В папку 'sorted_trash': {len(os.listdir(output_trash_dir))}")
        print(f"Оставлено (пропущено): {skipped_count}")
        print("----------------------")
        
    # ИЗМЕНЕНИЕ: Блок `finally` больше не нужен, `with` сам позаботится об удалении `temp_dir`.
    if args.debug:
        print("DEBUG: Временная папка была автоматически удалена.")


if __name__ == "__main__":
    main()
