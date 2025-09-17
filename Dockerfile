FROM python:3.11

# Définir le dossier de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .

# Installation des dépendances système
RUN apt-get update && apt-get install -y poppler-utils tesseract-ocr

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Téléchargement des modèles NLP et ressources NLTK
RUN python -m spacy download fr_dep_news_trf
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copier tout le reste
COPY . .

# Exposer le port de FastAPI
EXPOSE 8000

# Lancer le serveur
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
