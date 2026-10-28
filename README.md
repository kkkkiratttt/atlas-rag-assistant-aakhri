# Atlas RAG Assistant

A responsive, full-stack RAG website using **FastAPI, ChromaDB, Sentence Transformers, and Groq**.

## Features
- Upload PDF, TXT, Markdown, and DOCX files
- Local semantic embeddings with `all-MiniLM-L6-v2`
- Persistent ChromaDB vector search
- Groq-powered answers with retrieved source excerpts and [S1]-style citations
- Responsive dark UI with document indexing, follow-up chat, and optional password protection
- API keys remain on the server, not in browser JavaScript

## Run locally (Python 3.11 recommended)

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Set the environment variables:
- `GROQ_API_KEY`: create at https://console.groq.com/keys
- `APP_PASSWORD`: use a strong password (optional locally)
- `SESSION_SECRET`: long random secret
- `CHROMA_DB_PATH`: path to persistent vector storage, e.g. `./data/chroma_db`
- Optional `GROQ_MODEL` (default `openai/gpt-oss-20b`) and `GROQ_FALLBACK_MODELS` (default `openai/gpt-oss-120b`)

Start with `uvicorn app:app --host 0.0.0.0 --port 8000` and open http://localhost:8000.

## Deploy on Render

1. Open Render and create a **Blueprint** from this GitHub repository.
2. Review `render.yaml` and create the web service.
3. Add `GROQ_API_KEY` and a private, strong `APP_PASSWORD` to the service environment settings. The Blueprint generates `SESSION_SECRET`.
4. The persistent disk mounts at `/app/data`, and `CHROMA_DB_PATH` points to `/app/data/chroma_db` so indexed documents persist across deploys.
5. Wait for the service to become live, then open the service's `onrender.com` URL.

**Note:** The Blueprint uses a paid Render plan and persistent disk. Check the current pricing before creating the service. Never commit `.env`, API keys, passwords, or the ChromaDB data folder.
