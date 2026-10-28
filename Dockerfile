FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860 CHROMA_DB_PATH=/tmp/chroma_db XDG_CACHE_HOME=/tmp/cache
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY app.py rag_core.py ./
EXPOSE 7860
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
