# CVtensaHandSymbaaaaa

## Описание

`CVtensaHandSymbaaaaa` — это модульный portfolio-проект по компьютерному зрению для распознавания жестов рук в реальном времени. Текущий стабильный baseline уже умеет:

- захватывать видео с веб-камеры;
- детектировать руки через `MediaPipe Tasks Hand Landmarker`;
- отрисовывать landmarks и связи между точками;
- показывать видеопоток в Streamlit;
- записывать сырой датасет landmarks для последующего обучения модели.

На этапе 02 проект еще не распознает жесты, но уже готовит качественные и структурированные данные для следующего этапа обучения TensorFlow-модели.

## Стек технологий

- Python 3.11
- TensorFlow / Keras
- OpenCV
- MediaPipe Tasks
- NumPy
- Streamlit
- PyYAML

## Структура проекта

```text
CVtensaHandSymbaaaaa/
├── app/
│   ├── __init__.py
│   └── main.py
├── configs/
│   └── config.yaml
├── core/
│   ├── __init__.py
│   ├── camera.py
│   ├── config.py
│   ├── dataset_collector.py
│   ├── drawing_utils.py
│   ├── hand_tracker.py
│   ├── landmark_utils.py
│   └── logger.py
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── STAGE_01.md
│   └── stage_02.md
├── inference/
├── models/
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_dataset_collector.py
│   └── test_landmark_utils.py
├── training/
├── ui/
│   ├── __init__.py
│   └── streamlit_app.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Требования к окружению

- Python `3.11`
- рабочая веб-камера
- установленный `pip`
- файл модели `models/hand_landmarker.task`

## Создание виртуального окружения

Рекомендуемый способ установки:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run ui/streamlit_app.py
```

## Установка зависимостей

Если виртуальное окружение уже создано и активировано:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Добавление файла модели

Проект использует современный API MediaPipe Tasks и ожидает файл модели по пути:

```text
models/hand_landmarker.task
```

Если файла нет, приложение не падает, а показывает понятное предупреждение в интерфейсе и логах.

## Настройка этапа 02

Новый этап добавляет конфигурацию записи датасета в `configs/config.yaml`:

```yaml
dataset:
  raw_dir: data/raw
  index_file: data/raw/dataset_index.csv
  gesture_labels:
    - open_palm
    - fist
    - peace
    - thumbs_up
  capture_interval_ms: 250
  max_samples_per_label: 250
  require_single_hand: true
```

Что означает каждый параметр:

- `raw_dir` — каталог хранения JSON-семплов;
- `index_file` — CSV-индекс всех сохраненных sample;
- `gesture_labels` — разрешенные классы жестов;
- `capture_interval_ms` — минимальная пауза между сохранениями;
- `max_samples_per_label` — защитный лимит на класс;
- `require_single_hand` — запись только при одной детектированной руке.

## Как работает запись датасета

Каждый сохраненный sample содержит:

- исходные landmarks одной руки;
- нормализованные landmarks;
- плоский `feature_vector` длиной `63`;
- handedness;
- метаданные времени и label.

Нормализация выполняется так:

1. запястье (`landmark 0`) переносится в начало координат;
2. вся рука масштабируется по максимально удаленной точке;
3. результат становится устойчивее к сдвигу и изменению расстояния до камеры.

## Запуск

Основной способ запуска:

```bash
streamlit run ui/streamlit_app.py
```

Дополнительно доступен launcher:

```bash
python app/main.py
```

## Как записывать samples

1. Запустите видеопоток.
2. Выберите `Gesture label` в интерфейсе.
3. Нажмите `Включить запись`.
4. Удерживайте в кадре одну руку и меняйте позу, угол и дистанцию.
5. Следите за счетчиками сохраненных samples и статусом записи.

Сырые файлы будут складываться в `data/raw/<gesture_label>/`, а общий индекс — в `data/raw/dataset_index.csv`.

## Типовые ошибки и решения

### Не найдена модель MediaPipe

Симптом:

- в интерфейсе отображается предупреждение о том, что файл модели отсутствует.

Решение:

- положите `hand_landmarker.task` в каталог `models/`;
- убедитесь, что путь в `configs/config.yaml` совпадает с фактическим расположением файла.

### Камера не открывается

Симптом:

- в интерфейсе появляется ошибка подключения камеры.

Решение:

- проверьте, что камера не занята другим приложением;
- убедитесь, что `camera.device_id` в `configs/config.yaml` указан корректно;
- проверьте разрешения на доступ к камере в операционной системе.

### Samples не сохраняются

Симптом:

- запись включена, но счетчик не растет.

Решение:

- убедитесь, что в кадре находится только одна рука;
- проверьте, что выбран корректный `Gesture label`;
- дождитесь завершения паузы `capture_interval_ms`;
- проверьте права на запись в каталог `data/raw`.

## Текущее ограничение этапа

На этапе 02 жесты еще не классифицируются. Реализованы видеопоток, детекция рук, нормализация landmarks и сбор структурированного датасета для следующего этапа обучения.
