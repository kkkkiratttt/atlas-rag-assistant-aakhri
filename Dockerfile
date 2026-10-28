FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860 CHROMA_DB_PATH=/app/data/chroma_db HF_HOME=/app/data/huggingface
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/data/chroma_db /app/data/huggingface
EXPOSE 7860
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port $PORT"]
