# Morrigan API — image CPU (Phase 5, production)
#
# Single-stage : tous les wheels sont prebuilt (torch CPU, llama-cpp-python
# CPU via l'index abetlen dans requirements.txt) → aucune compilation native,
# pas besoin de multi-stage ni de toolchain.
#
# Le modèle GGUF et l'index ne sont PAS bakés : ils sont montés en volume
# (voir docker-compose.yml). L'image reste légère ; data/ est exclu via
# .dockerignore.
FROM python:3.12-slim

# torch CPU-only AVANT requirements : pip voit ensuite `torch>=2.0` déjà
# satisfait et n'embarque pas les libs CUDA (~plusieurs Go de gagnés).
# Morrigan est 100 % CPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

# Couche deps séparée du code pour le cache de build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exécution non-root.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Bind 0.0.0.0 (sinon 127.0.0.1 = injoignable depuis l'hôte) ; le conteneur
# écoute sur 8000 en interne (le mapping de port hôte est dans le compose).
# MORRIGAN_INDEX sert l'index Wikipedia persisté au démarrage.
# HF_HOME pointe sur un volume pour persister le cache MiniLM entre recreate.
ENV MORRIGAN_API_HOST=0.0.0.0 \
    MORRIGAN_API_PORT=8000 \
    MORRIGAN_INDEX=data/models/index_wiki \
    HF_HOME=/home/appuser/.cache/huggingface

EXPOSE 8000

CMD ["python", "-m", "interfaces.api"]
