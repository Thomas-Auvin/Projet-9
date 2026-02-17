# Projet 9 — POC RAG événements culturels

POC d’assistant intelligent basé sur un pipeline RAG (FAISS + Mistral) pour répondre à des questions sur des événements culturels.

## Setup
```bash
uv sync
```

## Variables d’environnement

Copier .env.example en .env et renseigner les variables (ne pas committer .env).

## Smoke check
```bash
uv run python -c "import faiss; import fastapi; import langchain"
```

## Tests
```bash
uv run pytest -q
```

