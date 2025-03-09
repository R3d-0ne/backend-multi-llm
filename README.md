# Service de Recherche Avancée avec Qdrant et LLM

Ce projet implémente un service de recherche avancée qui combine la recherche vectorielle avec Qdrant et l'utilisation de LLM (Large Language Models) pour améliorer les résultats de recherche et générer des réponses contextuelles.

## Architecture de la solution

Le service de recherche avancée implémente une approche hybride avec :

1. **Recherche sémantique par embeddings** : Utilisation des embeddings pour trouver des documents sémantiquement similaires à la requête
2. **Boost par mots-clés** : Amélioration des scores pour les documents contenant les mots-clés exacts de la requête
3. **Filtres sur métadonnées** : Possibilité de filtrer par divers attributs (dates, présence d'entités nommées, etc.)
4. **Réordonnancement par LLM** : Utilisation d'un LLM pour évaluer la pertinence réelle des documents trouvés
5. **Génération de réponses** : Création d'une réponse directe basée sur les documents les plus pertinents

## Composants principaux

### 1. Service de recherche (`search_service.py`)

Ce service orchestre tout le processus de recherche :
- Initialisation de la recherche vectorielle avec Qdrant
- Application des filtres sur les métadonnées
- Boost des résultats contenant des mots-clés de la requête
- Réordonnancement des résultats avec un LLM
- Génération de réponses basées sur les documents trouvés

### 2. API de recherche (`search.py`)

Expose deux endpoints pour la recherche :
- `/search/` (POST) : API complète avec filtres, réordonnancement LLM et génération de réponse
- `/search/simple` (GET) : Version simplifiée pour les requêtes rapides

## Améliorations techniques implémentées

1. **Recherche hybride** : Combinaison de similarité vectorielle et recherche par mots-clés pour améliorer la précision
2. **Réordonnancement adaptatif** : Le LLM évalue chaque document pour pondérer les résultats selon la pertinence réelle
3. **Filtrage multi-critères** : Support pour différents types de filtres (booléens, intervalles, recherche partielle)
4. **Génération de réponses contextuelles** : Le LLM utilise les documents les plus pertinents pour générer une réponse directe
5. **Optimisations de performance** : Récupération initiale d'un plus grand nombre de résultats pour permettre un meilleur réordonnancement

## Configuration

Le service nécessite les variables d'environnement suivantes :
- `QDRANT_HOST` et `QDRANT_PORT` : Configuration du serveur Qdrant
- `LLM_BASE_URL` : URL du service LLM (par défaut: http://localhost:11434)
- `LLM_MODEL` : Nom du modèle LLM à utiliser (par défaut: deepseek-coder)

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

## Évolutions futures

1. Implémentation de la recherche par "query reformulation" (le LLM réécrit la requête pour la rendre plus précise)
2. Ajout de la fonctionnalité de clustering des résultats pour regrouper les documents similaires
3. Support pour des requêtes en langage naturel plus complexes
4. Mise en cache des résultats fréquemment demandés 