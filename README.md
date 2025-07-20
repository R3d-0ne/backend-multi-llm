# Service de Recherche Avancée avec Qdrant et LLM

Ce projet implémente un service de recherche avancée qui combine la recherche vectorielle avec Qdrant et l'utilisation de LLM (Large Language Models) pour améliorer les résultats de recherche et générer des réponses contextuelles.

## Architecture de la solution

Le service de recherche avancée implémente une approche hybride avec :

1. **Recherche sémantique par embeddings** : Utilisation des embeddings pour trouver des documents sémantiquement similaires à la requête
2. **Boost par mots-clés** : Amélioration des scores pour les documents contenant les mots-clés exacts de la requête
3. **Filtres sur métadonnées** : Possibilité de filtrer par divers attributs (dates, présence d'entités nommées, etc.)
4. **Réordonnancement par LLM** : Utilisation d'un LLM pour évaluer la pertinence réelle des documents trouvés
5. **Génération de réponses** : Création d'une réponse directe basée sur les documents les plus pertinents

## Processus de recherche détaillé

Le processus de recherche implémente plusieurs étapes sophistiquées pour maximiser la pertinence des résultats :

### 1. Détection du type de recherche et préparation des vecteurs

- **Détection de la collection** : Le système détermine automatiquement si la collection cible est standard ou hybride (avec vecteurs nommés)
- **Génération des embeddings** :
  - **Vecteur dense** : Représentation vectorielle dense de la requête entière (768 dimensions)
  - **Vecteur d'entités** : Embedding spécifique des entités nommées dans la requête
  - **Vecteur sparse** : Représentation sparse (clairsemée) des mots-clés de la requête

### 2. Recherche vectorielle initiale

- **Pour les collections hybrides** (comme "documents") :
  - **Recherche multi-vectorielle** : Combine les vecteurs dense (60%), entités (30%) et sparse (10%)
  - **Détection automatique du format** : Utilise NamedVector de Qdrant pour la collection hybride
  - **Fallback intelligent** : Si la recherche avancée échoue, repli sur une recherche simple
- **Pour les collections standard** :
  - **Recherche simple** : Utilisation directe du vecteur dense

### 3. Filtrage des résultats (si applicable)

- **Filtres booléens** : Présence de tables, entités nommées, emails, etc.
- **Filtres de plage** : Dates d'upload, valeurs numériques
- **Filtres textuels** : Nom de fichier contenant des termes spécifiques

### 4. Boost des correspondances par mots-clés

- **Tokenisation** : La requête est décomposée en mots-clés significatifs
- **Analyse du contenu** : Chaque document est analysé pour la présence de ces mots-clés
- **Ajustement des scores** : +20% pour chaque mot-clé présent, plafonné à +100%

### 5. Réordonnancement par LLM

- **Évaluation individuelle** : Chaque document est soumis au LLM avec la requête
- **Notation de pertinence** : Le LLM attribue un score de 0 à 10 pour chaque document
- **Score combiné** : 70% score LLM + 30% score vectoriel original
- **Extraction robuste** : Capacité à extraire le score même en cas d'erreur de format

### 6. Génération de réponse contextuelle

- **Sélection des meilleurs documents** : Les 3 documents les plus pertinents sont retenus
- **Extraction intelligente du texte** : 
  - Détection multi-sources (cleaned_text, métadonnées, etc.)
  - Vérification de la qualité et du format du texte
  - Prétraitement pour limiter la taille à 2000 caractères par document
- **Construction du prompt contextuel** : Formatage structuré des documents pour le LLM
- **Appel API adaptatif** :
  - API chat pour les modèles Llama (comme llama3.1:8b)
  - API generate pour les autres modèles
- **Gestion robuste des réponses** :
  - Traitement des erreurs de parsing JSON
  - Extraction manuelle du contenu en cas d'échec
  - Messages d'erreur explicites et journalisation détaillée

## Support des collections hybrides

Le système prend en charge les collections hybrides de Qdrant qui permettent de combiner différents types de vecteurs :

- **Collection "documents"** : Configuration hybride avec vecteurs dense, entity et sparse
- **Autres collections** : Configuration standard avec un seul vecteur

Pour les collections hybrides, le système :
1. Détecte automatiquement le type de collection
2. Utilise le format NamedVector approprié
3. Combine intelligemment les résultats multi-vectoriels

## Optimisations techniques

### 1. Robustesse des appels API

- **Gestion avancée des erreurs** : Traitement des erreurs de connexion, timeout, et parsing
- **Extraction intelligente** : Capacité à extraire l'information même d'une réponse JSON malformée
- **Logging détaillé** : Journalisation complète pour faciliter le débogage

### 2. Adaptabilité aux modèles LLM

- **Détection automatique** : Adaptation du format selon le type de modèle (Llama, etc.)
- **Prompts optimisés** : Instructions spécifiques selon le modèle et la tâche
- **Paramètres ajustés** : Temperature de 0.1 pour maximiser la précision des réponses

### 3. Performance et scalabilité

- **Récupération optimisée** : Limite initiale plus élevée pour permettre un meilleur filtrage
- **Troncature intelligente** : Limitation de la taille des documents pour les appels LLM
- **Parallélisation potentielle** : Architecture permettant le traitement parallèle futur

## Composants principaux

### 1. Service de recherche (`search_service.py`)

Ce service orchestre tout le processus de recherche :
- Initialisation de la recherche vectorielle avec Qdrant
- Application des filtres sur les métadonnées
- Boost des résultats contenant des mots-clés de la requête
- Réordonnancement des résultats avec un LLM
- Génération de réponses basées sur les documents trouvés

### 2. Service Qdrant (`qdrant_service.py`)

Gère l'interaction avec la base de données vectorielle Qdrant :
- Détection automatique du type de collection (standard vs. hybride)
- Support pour les recherches hybrides avec vecteurs nommés
- Gestion robuste des différents formats de réponse

### 3. API de recherche (`search.py`)

Expose deux endpoints pour la recherche :
- `/search/` (POST) : API complète avec filtres, réordonnancement LLM et génération de réponse
- `/search/simple` (GET) : Version simplifiée pour les requêtes rapides
- `/search/internal/` (POST) : Recherche avec génération de réponse contextuelle

## Configuration

Le service nécessite les variables d'environnement suivantes :
- `QDRANT_HOST` et `QDRANT_PORT` : Configuration du serveur Qdrant
- `LLM_BASE_URL` : URL du service LLM (par défaut: http://host.docker.internal:11434)
- `LLM_MODEL` : Nom du modèle LLM à utiliser (par défaut: llama3.1:8b)

## Utilisation

### Exemple de requête POST

```json
POST /search/
{
  "query": "Documents contenant des informations sur les contrats d'assurance",
  "limit": 5,
  "filters": {
    "has_tables": true,
    "upload_date_range": {
      "start": "2023-01-01T00:00:00",
      "end": "2023-12-31T23:59:59"
    }
  },
  "use_llm_reranking": true,
  "boost_keywords": true,
  "generate_answer": true
}
```

### Exemple de requête GET simplifiée

```
GET /search/simple?q=contrats%20assurance&limit=5&generate_answer=true
```

### Exemple de recherche interne avec génération de réponse

```json
POST /search/internal/
{
  "query": "Quelles sont les clauses importantes dans les contrats d'assurance?",
  "limit": 5,
  "filters": {
    "has_tables": true
  }
}
```

## Installation et exécution

1. Installer les dépendances : `pip install -r requirements.txt`
2. Configurer les variables d'environnement dans un fichier `.env`
3. Lancer le serveur : `uvicorn app.main:app --reload`

## Paramétrage avancé

Le service permet de configurer :
- Les prompts système utilisés pour le réordonnancement et la génération de réponses
- La pondération entre scores de similarité vectorielle et scores LLM
- Le facteur de boost pour les correspondances de mots-clés
- Le nombre de documents à utiliser pour la génération de réponses

## Bonnes pratiques implémentées

1. **Traçabilité complète** : Conservation des scores originaux et des scores LLM pour chaque document
2. **Gestion d'erreurs robuste** : Fallback vers la recherche vectorielle en cas d'échec du LLM
3. **Optimisation des prompts** : Instructions précises pour le LLM afin d'obtenir des résultats cohérents
4. **Documentation complète** : Tous les paramètres et méthodes sont documentés
5. **Logs détaillés** : Journalisation extensive pour faciliter le débogage

## Évolutions futures

1. Implémentation de la recherche par "query reformulation" (le LLM réécrit la requête pour la rendre plus précise)
2. Ajout de la fonctionnalité de clustering des résultats pour regrouper les documents similaires
3. Support pour des requêtes en langage naturel plus complexes
4. Mise en cache des résultats fréquemment demandés
5. Parallélisation du réordonnancement LLM pour améliorer les performances 