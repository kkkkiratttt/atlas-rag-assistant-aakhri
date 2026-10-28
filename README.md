# Atlas RAG Assistant

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/kkkkiratttt/atlas-rag-assistant-final-)

A responsive, full-stack RAG website using **FastAPI, ChromaDB, Sentence Transformers, and Groq**.

## Features
- Upload PDF, TXT, Markdown, and DOCX files
- Local semantic embeddings with `all-MiniLM-L6-v2`
- Persistent ChromaDB vector search
- Groq-powered answers with source excerpts and [S1]-style citations
- Responsive dark UI with document indexing, follow-up chat, and optional password protection
- API keys remain on the server, not in browser JavaScript

## Run locally (Python 3.11 recommended)

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Set these environment variables:
- `GROQ_API_KEY`: create at https://console.groq.com/keys
- `APP_PASSWORD`: use a strong private password (optional locally)
- `SESSION_SECRET`: long random secret
- `CHROMA_DB_PATH`: persistent local path, e.g. `./data/chroma_db`
- Optional `GROQ_MODEL` (default `openai/gpt-oss-20b`) and `GROQ_FALLBACK_MODELS` (default `openai/gpt-oss-120b`)

Run `uvicorn app:app --host 0.0.0.0 --port 8000` and open http://localhost:8000.

## Deploy on Render

1. Click the **Deploy to Render** button above, sign in to Render, and grant access to this private GitHub repository if prompted.
2. Review the Blueprint settings. This repository's `render.yaml` configures the Docker service, health check, Groq model, and persistent ChromaDB directory.
3. Enter `GROQ_API_KEY` and a strong private `APP_PASSWORD` as secrets. The Blueprint generates `SESSION_SECRET`.
4. Confirm the plan and any charges, then create the service. Persistent disks require a paid Render web service; verify current pricing before proceeding.
5. After the first deploy succeeds, open the Render-provided `onrender.com` URL.

The first startup downloads the Sentence Transformers embedding model and may take several minutes. Indexed documents persist under `/app/data/chroma_db`.

**Never commit `.env`, API keys, passwords, or the ChromaDB data folder.**
