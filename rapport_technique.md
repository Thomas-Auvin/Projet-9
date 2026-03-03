# Rapport Technique – Assistant intelligent de recommandation d’événements culturels

## 1. Objectifs du projet

* **Contexte :** Mission confiée par Puls-Events pour moderniser l'accès aux événements culturels via un assistant conversationnel intelligent.
* **Problématique :** Un système RAG (Retrieval-Augmented Generation) permet d'ancrer les réponses de l'IA sur des données réelles (OpenAgenda), garantissant des informations fiables et limitant les hallucinations.
* **Objectif du POC :** Démontrer la viabilité d'un moteur RAG propriétaire performant, industrialisé et évaluable.
* **Périmètre :** Événements de la base OpenAgenda (Bordeaux/Nouvelle-Aquitaine).

## 2. Architecture et Industrialisation

* **Schéma global :** Un pipeline RAG hybride associant briques standardisées et orchestration personnalisée.
* **Ingestion :** Script de nettoyage et conversion en format `Document` LangChain.
* **Moteur (Engine) :** Logique métier centralisée dans `rag/moteur/engine.py`.
* **Recherche :** Similarité sémantique via **FAISS** (wrapper LangChain).
* **Génération :** Appel direct via le SDK `mistralai`.


* **Déploiement & Docker :** L'API est packagée dans un container Docker pour garantir l'isomorphisme entre les environnements. Les volumes sont utilisés pour persister les fichiers d'index (`artifacts/`) et les logs d'exécution.
* **CI/CD :** Utilisation de **GitHub Actions** pour automatiser la qualité :
* Linting et formatage via `ruff`.
* Tests unitaires et d'intégration via `pytest`.
* Build et validation de l'image Docker.



## 3. Préparation et vectorisation des données

* **Source de données :** Export OpenAgenda (`events_processed_geo.csv`).
* **Formatage :** Utilisation de `langchain_core.documents.Document`. Cette structure couple le texte (`page_content`) aux métadonnées critiques (`event_id`, titre, URL).
* **Chunking :** Stratégie **"Un événement = Un document"**.
* **Avantage :** Retrieval stable et métadonnées intactes (pas de perte de contexte sur le lieu ou la date).
* **Compromis :** Documents parfois longs (trade-off sur la fenêtre de contexte).


* **Embedding :** Modèle `mistral-embed` (dim 1024) avec stockage FAISS on-disk.

## 4. Choix du modèle NLP et Orchestration

* **Modèle :** `mistral-small-latest` (choisi pour son rapport qualité/raisonnement/coût).
* **Orchestration Custom :** Gérée dans le `RagEngine`. Elle inclut le filtrage temporel post-retrieval et la gestion des variantes de prompts (`P9_PROMPT_VARIANT`).
* **Viabilité Économique (Hypothèses) :** * Basé sur un volume de **5 000 requêtes/mois** (contexte moyen de 1500 tokens).
* Modèle Mistral Small (~0,30$/1M tokens).
* **Coût estimé : < 3 € / mois**, validant la scalabilité financière du projet.



## 5. Construction de la base vectorielle

* **FAISS (VectorStore LangChain) :** Utilisation pour l'API `similarity_search` et le retour d'objets `Document`.
* **Persistance :** Dossier `artifacts/faiss_index_mistral` (serialization `.faiss` et `.pkl`).
* **Métadonnées :** Chaque vecteur pointe vers un `event_id`, garantissant la traçabilité de la source.

## 6. API et endpoints exposés

* **Framework :** FastAPI (Python 3.13, géré par `uv`).
* **Endpoints clés :**
* `POST /ask` : Entrée principale (Question -> Context -> Answer).
* `POST /rebuild` : Mise à jour de l'index via `X-Rebuild-Token`.
* `GET /history` : Historique des **requêtes/réponses (turns)** en mémoire (turn_id, question, réponse, sources, rating).
* `POST /feedback` : Collecte des feedbacks par `turn_id`.



## 7. Évaluation du système

Le système est évalué en "Single Execution" (une seule passe) pour refléter l'usage réel :

* **Golden Accuracy (Dataset `golden.jsonl`) :** * Métriques : **Hit@k**, **Recall**, **MRR**, et **Precision@k**.
* Validation stricte : Correspondance entre `sources[].id` et l'ID de l'événement attendu (`event_id`).


* **RAGAS :** Mesure de la *Faithfulness* et de l'intérêt de la réponse générée.

## 8. Recommandations et perspectives

* **Points forts :** Pipeline industrialisé (Docker/CI), maîtrise totale de l'orchestration, et faible coût.
* **Limites :** Recherche purement sémantique ; pourrait bénéficier d'une hybridation avec des filtres SQL/Metadata.
* **Améliorations :** Implémentation d'un **Reranker** et passage à une base vectorielle managée (ex: pgvector) pour la production.

## 9. Organisation du dépôt GitHub

* **`app/`** : Serveur FastAPI et routes.
* **`rag/moteur/`** :
* `engine.py` : Orchestrateur principal.
* `prompting.py` : Gestion des templates de prompts.
* `rag_chain.py` : Utilitaires `Document` et client Mistral avec gestion des **retries/backoff (429)**.


* **`evals/`** : Scripts RAGAS et Golden Accuracy.
* **`artifacts/`** : Index FAISS persistant.
* **`tests/`** : Suite Pytest (CI).
