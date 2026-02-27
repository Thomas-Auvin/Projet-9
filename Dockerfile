FROM python:3.13-slim

# évite de créer des .pyc, logs plus clairs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# installer uv
RUN pip install --no-cache-dir uv

# copier uniquement les fichiers de dépendances pour cache docker
COPY pyproject.toml uv.lock /app/

# installer deps
RUN uv sync --frozen

# copier le code
COPY app /app/app
COPY rag /app/rag
COPY scripts /app/scripts
COPY project_paths.py /app/project_paths.py

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
