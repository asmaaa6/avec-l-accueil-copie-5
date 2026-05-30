FROM python:3.12-slim

# 1. Installation de Tesseract OCR et de ses dépendances système
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fra \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Installation des dépendances Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copie du code source
COPY . /app

# 4. Configuration du port pour Render
EXPOSE 10000
ENV PORT=10000

# 5. Commande de démarrage (sans les crochets pour que la variable $PORT fonctionne)
CMD gunicorn app:app --bind 0.0.0.0:$PORT