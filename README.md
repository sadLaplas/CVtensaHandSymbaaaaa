# CVtensaHandSymbaaaaa

## Описание

`CVtensaHandSymbaaaaa` — это базовый production-ready каркас проекта компьютерного зрения для работы с жестами рук. На текущем этапе приложение захватывает изображение с веб-камеры, определяет руки через MediaPipe Tasks Hand Landmarker и отображает landmarks в Streamlit-интерфейсе.

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
│   └── main.py
├── configs/
│   └── config.yaml
├── core/
│   ├── camera.py
│   ├── config.py
│   ├── drawing_utils.py
│   ├── hand_tracker.py
│   └── logger.py
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   └── STAGE_01.md
├── inference/
├── models/
├── tests/
│   └── test_config.py
├── training/
├── ui/
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

Если файла нет, приложение не падает, а показывает понятное предупреждение в интерфейсе и логах. Модель можно подготовить заранее и положить в каталог `models/`.

## Запуск

Основной способ запуска:

```bash
streamlit run ui/streamlit_app.py
```

Дополнительно доступен launcher:

```bash
python app/main.py
```

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

### Не устанавливается MediaPipe или OpenCV

Симптом:

- приложение сообщает об отсутствии зависимостей.

Решение:

- используйте Python `3.11`;
- обновите `pip`;
- повторно установите зависимости из `requirements.txt`.

## Текущее ограничение этапа

На этапе 01 жесты еще не распознаются. Реализованы только видеопоток, детекция рук и визуализация landmarks.
