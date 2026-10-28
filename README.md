# Atlas RAG Assistant — Free Demo

Responsive document RAG website using FastAPI, ChromaDB, Chroma's ONNX MiniLM embedding function, and Groq.

## Deploy free on Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/kkkkiratttt/atlas-rag-assistant-final-)

1. Click the deploy button and authorize Render to access this GitHub repository.
2. Choose the **Free** web service plan if prompted.
3. Add `GROQ_API_KEY` from https://console.groq.com/keys.
4. Set a strong, private `APP_PASSWORD`. Do not share this password if you don't want other people using your Groq requests.
5. Keep the generated `SESSION_SECRET`. Wait for deployment and open the Render URL.

## Free-tier limitations

- Render Free provides 512 MB RAM and no persistent disk.
- The service sleeps after 15 minutes with no inbound traffic; the next visit may take about a minute to wake up.
- **Uploaded files and ChromaDB index are temporary.** They can be lost when the service sleeps, restarts, or redeploys. You may need to re-upload and re-index your files after a cold start.
- Chroma's default ONNX `all-MiniLM-L6-v2` embedding function avoids loading the full PyTorch/Sentence Transformers stack. The model still needs to download on first use.
- Intended for a class project/demo, not permanent multi-user storage.

## Run locally (Python 3.11 recommended)

```bash
pip install -r requirements.txt
```

Set environment variables (PowerShell example):

```powershell
$env:GROQ_API_KEY="your-groq-key"
$env:APP_PASSWORD="your-private-password"
$env:SESSION_SECRET="a-long-random-secret"
$env:CHROMA_DB_PATH="./data/chroma_db"
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. Upload PDF, TXT, Markdown, or DOCX files, click **Index documents**, and ask questions. The assistant answers with source labels and expandable excerpts.

Never commit API keys, passwords, `.env`, or ChromaDB data.
