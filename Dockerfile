FROM python:3.11

# Définir le dossier de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le reste
COPY . .

# Exposer le port de FastAPI
EXPOSE 8000

# Lancer le serveur
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
