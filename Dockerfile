# Официальный образ Python 3.10
FROM python:3.10-slim

# Обновление пакетов и установка зависимостей системы для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Создание и установка рабочей директории внутри контейнера
WORKDIR /app

# Копирование файла зависимостей
COPY requirements.txt .

# Установка Python-зависимостей без кэша для уменьшения размера образа
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода проекта
COPY . .

# Открытие порта 8501 для доступа к Streamlit
EXPOSE 8501

# Переменные окружения для правильной работы Streamlit в Docker
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Команда запуска приложения
CMD ["streamlit", "run", "app/main.py"]
