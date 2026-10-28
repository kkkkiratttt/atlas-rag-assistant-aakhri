# Atlas RAG Assistant

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/kkkkiratttt/atlas-rag-assistant-final-)

A responsive, full-stack RAG website using **FastAPI, ChromaDB, Sentence Transformers, and Groq**.

## Features
- Upload PDF, TXT, Markdown, and DOCX files
- Local semantic embeddings with `all-MiniLM-L6-v2`
- Persistent ChromaDB vector search
- Groq-powered answers with source excerpts and [S1]-style citations
- Responsive dark UI, document indexing, follow-up chat, and optional password protection
- Groq API key is only used server-side

## Run locally (Python 3.11 recommended)

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Set the environment variables:
- `GROQ_API_KEY`: create at https://console.groq.com/keys
- `APP_PASSWORD`: strong private password (optional locally)
- `SESSION_SECRET`: long random secret
- `CHROMA_DB_PATH`: persistent local path, e.g. `./data/chroma_db`
- Optional `GROQ_MODEL` (default `openai/gpt-oss-20b`) and `GROQ_FALLBACK_MODELS` (default `openai/gpt-oss-120b`)

Start with `uvicorn app:app --host 0.0.0.0 --port 8000`; open http://localhost:8000.

## Deploy on Render

1. Click the **Deploy to Render** button above and sign in.
2. Grant Render access to this private GitHub repo if asked, then review the Blueprint.
3. Enter `GROQ_API_KEY` and a strong private `APP_PASSWORD` as secrets. The Blueprint generates `SESSION_SECRET`.
4. Confirm the plan and charges before creating the service. This configuration uses Render's Standard service plan and a 1 GB persistent disk because the embedding stack needs more memory and the ChromaDB index must survive restarts. Check https://render.com/pricing before approving.
5. Wait for the first deploy; the embedding model downloads on first startup. Open the `onrender.com` URL when the service is live.

ChromaDB is stored at `/app/data/chroma_db`; the Hugging Face model cache is stored under `/app/data/huggingface`. Both are under the persistent disk.

**Never commit `.env`, API keys, passwords, or the ChromaDB data folder.**
