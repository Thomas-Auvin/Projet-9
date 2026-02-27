# Projet 9 — Puls-Events RAG POC

POC d’un système **RAG** (Retrieval-Augmented Generation) pour recommander des événements (base OpenAgenda) via :
- **FAISS** pour la recherche sémantique (vector store),
- **Mistral** pour les embeddings et la génération,
- **FastAPI** pour l’exposition en API REST.

> ⚠️ **Pré-requis clé :** le démarrage nécessite une clé **Mistral** (`MISTRAL_API_KEY`) fournie via un fichier `.env` basé sur `.env.example`.

---

## Sommaire
- [Quickstart (Docker Compose)](#quickstart-docker-compose)
- [Endpoints API](#endpoints-api)
- [Rebuild index (2 options)](#rebuild-index-2-options)
- [Configuration (variables d’environnement)](#configuration-variables-denvironnement)
- [Développement local (uv)](#développement-local-uv)
- [Tests & Qualité](#tests--qualité)
- [Évaluation (RAGAS)](#évaluation-ragas)
- [Architecture](#architecture)
- [Dépannage](#dépannage)

---

## Quickstart (Docker Compose)

### 1) Configurer l’environnement
Copier `.env.example` en `.env` puis renseigner :
- `MISTRAL_API_KEY` (**obligatoire**)
- (optionnel) `REBUILD_TOKEN` si tu veux activer l’endpoint `/rebuild`

### 2) Lancer l’API
Pour lancer l'API, il est conseillé d'utiliser docker.
Premier lancement de l'API :

```bash
docker compose up --build
```

Pour tout autre lancement sans modification du code :
```bash
docker compose up
```

Si vous souhaitez la lancer en local utilisez la commande suivante : 
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Description de l'API

## Endpoints API

L’API est définie dans app/main.py :

GET /health → état OK 
POST /ask → pose une question (RAG) 
GET /history → historique des derniers tours (taille configurable) 
POST /feedback → vote sur un turn_id (sert au suivi/itérations) 
POST /rebuild → reconstruit l’index (protégé par token) 

Pour poser une question, utilisez POST /ask en remplissant le JSON minimal :
```json
{"question": "Que faire ce week-end a Bordeaux ?"}
```
Option : k (top-k retrieval) peut être fourni dans le body ; sinon l’API utilise un défaut (voir P9_K_DEFAULT). 

La réponse inclut notamment answer + sources (IDs, titres, urls, dates…) et des métadonnées (modèle, k, etc.).

POST /rebuild (protégé)

Si REBUILD_TOKEN est absent → 403 REBUILD_TOKEN not set; rebuild disabled. 
Si token invalide → 403 Invalid rebuild token. 
Le token est enregistré dans .env :
X-Rebuild-Token: <REBUILD_TOKEN>

### Configuration (variables d’environnement)

L’API lit ces variables au démarrage : 

MISTRAL_API_KEY (obligatoire ; utilisé par le moteur RAG)
P9_INDEX_DIR (défaut : artifacts/faiss_index_mistral) 
P9_PROCESSED_CSV (défaut : data/processed/events_processed_geo.csv) 
P9_K_DEFAULT (défaut : 6) 
MISTRAL_CHAT_MODEL (défaut : mistral-small-latest) 
MISTRAL_EMBED_MODEL (défaut : mistral-embed) 
P9_HISTORY_SIZE (défaut : 10) 
P9_HISTORY_PROMPT_TURNS (défaut : 3) 
REBUILD_TOKEN (optionnel ; active /rebuild) 

Pour remplis la clé mistral, il est nécessaire de s'inscrire et de demander une clé sur l'application Mistral à l'adresse suivante : https://admin.mistral.ai/organization/api-keys

### Tests & Qualité

Pour toutes modifications du code, la branche main est protégé et une demande de pull request est nécessaire. Elle passera nécessairement par un CI. 
Afin de vérifier la qualité du code, vous pouvez lancer les tests en local : 

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

### Évaluation (RAGAS)

Un script d’évaluation existe dans evals/. Pour le lancer il faut faire la commande suivante :

```bash
uv run python evals/run_ragas.py --k 6 --limit 20
```

Les résultats (CSV) sont enregistrés dans evals/experiments/.