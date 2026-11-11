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
