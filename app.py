import os,secrets,threading
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI,File,HTTPException,Request,UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel,Field
from starlette.middleware.sessions import SessionMiddleware
BASE=Path(__file__).resolve().parent
load_dotenv(BASE/".env")
from rag_core import RAGAssistant,SUPPORTED_EXTENSIONS
DB_PATH=os.getenv("CHROMA_DB_PATH",str(BASE/"data"/"chroma_db"))
PASSWORD=os.getenv("APP_PASSWORD","").strip()
app=FastAPI(title="Atlas RAG Assistant")
app.add_middleware(SessionMiddleware,secret_key=os.getenv("SESSION_SECRET","").strip() or secrets.token_urlsafe(32),same_site="lax",https_only=os.getenv("COOKIE_HTTPS_ONLY","false").lower()=="true",max_age=43200)
_rag=None; _lock=threading.Lock()
def rag():
 global _rag
 if _rag is None:
  with _lock:
   if _rag is None:
    Path(DB_PATH).mkdir(parents=True,exist_ok=True); _rag=RAGAssistant(DB_PATH)
 return _rag
def auth(req):
 if PASSWORD and not req.session.get("ok"): raise HTTPException(401,"Please sign in to Atlas.")
class Login(BaseModel): password:str=Field(max_length=512)
class Chat(BaseModel):
 message:str=Field(min_length=1,max_length=8000)
 history:list[dict[str,str]]=Field(default_factory=list,max_length=12)
 top_k:int=Field(default=4,ge=1,le=10)
PAGE=r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Atlas — Document Intelligence</title><style>
:root{color-scheme:dark;--bg:#111018;--panel:#191822;--line:#32303f;--muted:#a7a4b6;--text:#f4f1fc;--violet:#a78bfa;--green:#5ee0b2}*{box-sizing:border-box}body{margin:0;background:radial-gradient(850px 500px at 90% 0%,#31234c,transparent 65%),var(--bg);font:15px/1.5 Inter,system-ui,sans-serif;color:var(--text)}button,input,textarea{font:inherit}button{border:1px solid var(--line);border-radius:12px;padding:10px 13px;background:#24222f;color:var(--text);cursor:pointer}button:hover{border-color:var(--violet)}button.primary{background:linear-gradient(135deg,#8b5cf6,#635bfa);border:0;font-weight:700}.layout{max-width:1400px;margin:auto;padding:20px;min-height:100vh;display:grid;grid-template-columns:300px minmax(0,1fr);gap:18px}.side,.main{background:#181720eF;border:1px solid #ffffff12;border-radius:20px;overflow:hidden}.side{padding:20px;display:flex;flex-direction:column;gap:18px;height:calc(100vh - 40px);position:sticky;top:20px}.brand{display:flex;gap:11px;align-items:center}.logo{width:43px;height:43px;border-radius:13px;background:linear-gradient(145deg,#b39aff,#6847ce);display:grid;place-items:center;font-size:23px}.brand h1{margin:0;font-size:18px}.brand small{color:var(--muted);font-size:10px;letter-spacing:1.3px}.tag{font-size:10px;color:#d1c1ff;border:1px solid #7862b8;border-radius:30px;padding:4px 9px;width:max-content;letter-spacing:.7px}.head{display:flex;justify-content:space-between;align-items:center;font-weight:750;font-size:13px;margin-bottom:10px}.muted{color:var(--muted);font-size:12px}.drop{border:1px dashed #60596f;border-radius:15px;padding:16px;text-align:center;background:#fff2}.drop.drag{border-color:var(--violet);background:#a78bfa10}.drop p{font-size:11px;color:var(--muted);margin:6px}.actions{display:flex;gap:8px;margin-top:9px}.actions button:first-child{flex:1}.list{display:flex;flex-direction:column;gap:7px;overflow:auto;min-height:30px}.doc{display:flex;gap:9px;padding:9px;border:1px solid var(--line);border-radius:11px}.fileicon{background:#9b7af122;color:#d2c4ff;border-radius:8px;padding:8px 5px;font-size:10px;font-weight:800;align-self:start}.filename{font-size:11px;overflow-wrap:anywhere}.meta{font-size:10px;color:var(--muted)}.status{background:#5ee0b20d;color:var(--green);border-radius:10px;padding:10px;font-size:11px}.bottom{margin-top:auto;padding-top:12px;border-top:1px solid var(--line);font-size:11px;color:var(--muted);line-height:1.9}.main{display:flex;flex-direction:column;min-height:calc(100vh - 40px)}.top{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:20px 23px;border-bottom:1px solid var(--line)}.top h2{font-size:16px;margin:0}.online{color:var(--muted);font-size:11px}.dot{display:inline-block;background:var(--green);width:7px;height:7px;border-radius:50%;margin-right:6px}.chat{padding:23px;display:flex;flex-direction:column;gap:18px;flex:1;overflow:auto}.welcome{text-align:center;max-width:630px;margin:auto;padding:35px 8px}.welcome h2{font-size:29px;letter-spacing:-1px;margin:12px}.grad{background:linear-gradient(90deg,#d8c8ff,#9b83ff,#75d9d1);color:transparent;background-clip:text}.welcome p{color:var(--muted);margin:0 auto 22px;max-width:480px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:9px;text-align:left}.suggest{text-align:left;font-size:11px;background:#ffffff04}.suggest b{display:block;font-size:12px;margin-bottom:4px}.msg{max-width:850px;width:100%;margin:auto;display:flex;gap:11px;align-items:flex-start}.avatar{width:33px;height:33px;flex:0 0 33px;border-radius:11px;display:grid;place-items:center;background:#9b7af122;color:#d0c0ff}.bubble{flex:1;min-width:0}.who{font-size:11px;color:var(--muted);margin-bottom:5px}.answer{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.75}.sources{margin-top:12px;border:1px solid var(--line);border-radius:12px;overflow:hidden}.sources summary{cursor:pointer;padding:10px;font-size:11px;color:#d1c1ff}.source{padding:10px;border-top:1px solid var(--line);font-size:11px}.source p{white-space:pre-wrap;color:var(--muted);margin:5px 0}.composewrap{padding:16px 20px 20px;border-top:1px solid var(--line)}.compose{max-width:870px;margin:auto;display:flex;gap:8px;align-items:flex-end;border:1px solid #504958;background:#201e29;padding:8px;border-radius:15px}.compose textarea{flex:1;background:transparent;color:var(--text);border:0;outline:0;resize:vertical;min-height:42px;max-height:140px;padding:9px}.send{height:42px;min-width:44px}.hint{text-align:center;color:#7e7a8b;font-size:10px;margin-top:8px}.hidden{display:none!important}.modal{position:fixed;inset:0;background:#05040be0;display:grid;place-items:center;padding:20px}.card{background:var(--panel);padding:25px;border:1px solid var(--line);border-radius:18px;width:min(400px,100%)}.card h2{margin:8px 0}.card p{font-size:12px;color:var(--muted)}.card input{width:100%;padding:12px;margin:8px 0;background:#111018;border:1px solid var(--line);border-radius:10px;color:var(--text)}.card button{width:100%}@media(max-width:840px){.layout{grid-template-columns:1fr;padding:10px;gap:10px}.side{position:static;height:auto}.main{min-height:70vh}.list{max-height:150px}.bottom{margin-top:0}}@media(max-width:520px){.layout{padding:7px}.side{padding:15px}.top,.chat{padding:14px}.grid{grid-template-columns:1fr}.welcome h2{font-size:24px}.composewrap{padding:12px}.online{font-size:0}.dot{font-size:12px}}
</style></head><body><div class="layout"><aside class="side"><div class="brand"><div class="logo">✦</div><div><h1>Atlas</h1><small>DOCUMENT INTELLIGENCE</small></div></div><div class="tag">RAG WORKSPACE</div><section><div class="head"><span>Knowledge base</span><span class="muted" id="count">0 chunks</span></div><div class="drop" id="drop"><div style="font-size:25px">⇧</div><b style="font-size:12px">Add your documents</b><p>PDF · TXT · Markdown · DOCX<br>Up to 20 MB per file</p><input id="files" type="file" multiple accept=".pdf,.txt,.md,.docx" hidden><button id="choose">Choose files</button></div><div class="actions"><button class="primary" id="index">Index documents</button><button id="reset">Reset</button></div><div id="uploadStatus" class="muted" style="margin-top:8px">No pending uploads</div></section><section style="flex:1;min-height:0;display:flex;flex-direction:column"><div class="head"><span>Indexed files</span><button id="refresh" style="font-size:10px;padding:4px 8px">Refresh</button></div><div class="list" id="docs"><span class="muted">Loading…</span></div></section><div class="status" id="server">Checking service…</div><div class="bottom">Vector search · ChromaDB<br>Embeddings · Sentence Transformers<br>Answer generation · Groq API<br><span style="color:#827d90">API credentials stay server-side.</span></div></aside><main class="main"><header class="top"><div><h2>Ask your knowledge base</h2><div class="muted">Grounded answers with source excerpts.</div></div><div class="online"><i class="dot"></i>ONLINE</div><button id="logout" class="hidden">Sign out</button></header><div id="chat" class="chat"><div class="welcome" id="welcome"><div style="font-size:38px">✧</div><div class="tag" style="margin:auto">YOUR DOCUMENTS, UNDERSTOOD</div><h2>Knowledge, without the <span class="grad">digging.</span></h2><p>Upload notes, reports, guides, or policies. Atlas will find relevant passages and cite its sources.</p><div class="grid"><button class="suggest" data-q="What are the main topics covered in my documents?"><b>✳ Explore my documents</b>Summarize the key topics</button><button class="suggest" data-q="Explain the most relevant process step by step."><b>⌁ Explain a process</b>Turn instructions into steps</button><button class="suggest" data-q="What deadlines or time limits are mentioned?"><b>◷ Find important details</b>Look for dates and limits</button><button class="suggest" data-q="What information is missing or unclear?"><b>◎ Spot information gaps</b>Identify unclear details</button></div></div></div><div class="composewrap"><form id="form" class="compose"><textarea id="question" placeholder="Ask a question about your documents…" rows="1" required></textarea><button class="primary send" id="send">↑</button></form><div class="hint">Check citations and important details against the original files.</div></div></main></div><div class="modal hidden" id="modal"><form class="card" id="login"><div class="logo">✦</div><h2>Welcome to Atlas</h2><p>This workspace is password protected. Enter the workspace password.</p><input id="password" type="password" placeholder="Workspace password" required><button class="primary">Unlock workspace</button><div id="loginError" style="color:#ff9bac;font-size:12px;margin-top:8px"></div></form></div>
<script>
const $=id=>document.getElementById(id);let files=[],history=[],busy=false;function esc(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
async function api(path,opts={}){let r=await fetch(path,opts),d={};try{d=await r.json()}catch{}if(!r.ok){if(r.status===401&&path!=="/api/login")$("modal").classList.remove("hidden");throw Error(d.detail||"Request failed "+r.status)}return d}
function showDocs(ds){$("docs").innerHTML=ds.length?ds.map(d=>'<div class="doc"><div class="fileicon">'+esc((d.source.split(".").pop()||"FILE").toUpperCase())+'</div><div><div class="filename">'+esc(d.source)+'</div><div class="meta">'+d.chunks+' chunks</div></div></div>').join(""):'<span class="muted">No documents indexed yet.</span>'}
async function refresh(){try{let d=await api("/api/documents");showDocs(d.documents||[]);$("count").textContent=(d.total_chunks||0)+" chunks"}catch(e){$("docs").textContent=e.message}}
async function init(){try{let s=await api("/api/session");if(s.auth_required&&!s.authenticated){$("modal").classList.remove("hidden");return}if(s.auth_required)$("logout").classList.remove("hidden");await Promise.all([refresh(),api("/api/health").then(h=>{$("server").textContent=h.groq_configured?"● Groq configured":"● Add GROQ_API_KEY in hosting settings";$("server").style.color=h.groq_configured?"var(--green)":"#ffc56d"})])}catch(e){$("server").textContent=e.message}}
function msg(role,text,sources=[]){$("welcome")?.remove();let node=document.createElement("div");node.className="msg";node.innerHTML='<div class="avatar">'+(role==="user"?"Y":"✦")+'</div><div class="bubble"><div class="who">'+(role==="user"?"You":"Atlas")+'</div><div class="answer">'+esc(text)+'</div></div>';$("chat").append(node);if(sources.length){let d=document.createElement("details");d.className="sources";d.innerHTML='<summary>View '+sources.length+' retrieved sources</summary>'+sources.map(s=>'<div class="source"><b>['+esc(s.id)+'] '+esc(s.source)+'</b> · chunk '+esc(s.chunk)+'<p>'+esc(s.excerpt)+'</p></div>').join("");node.querySelector(".bubble").append(d)}$("chat").scrollTop=$("chat").scrollHeight;return node}
function working(v){busy=v;$("send").disabled=v;$("index").disabled=v;$("send").textContent=v?"…":"↑"}
$("choose").onclick=()=>$("files").click();$("files").onchange=e=>{files=[...e.target.files];$("uploadStatus").textContent=files.length?files.length+" file(s) selected":"No pending uploads"};$("drop").ondragover=e=>{e.preventDefault();$("drop").classList.add("drag")};$("drop").ondragleave=()=> $("drop").classList.remove("drag");$("drop").ondrop=e=>{e.preventDefault();$("drop").classList.remove("drag");files=[...e.dataTransfer.files];$("uploadStatus").textContent=files.length?files.length+" file(s) selected":"No pending uploads"};
$("index").onclick=async()=>{if(busy)return;if(!files.length){$("uploadStatus").textContent="Choose a file first.";return}working(true);$("uploadStatus").textContent="Extracting, embedding and indexing…";try{let f=new FormData();files.forEach(x=>f.append("files",x));let d=await api("/api/documents/index",{method:"POST",body:f});$("uploadStatus").textContent="Indexed "+(d.indexed_files||[]).length+" file(s), "+(d.total_chunks||0)+" chunks.";if(d.errors?.length)$("uploadStatus").textContent+=" Error: "+d.errors.map(x=>x.source+": "+x.error).join("; ");files=[];$("files").value="";await refresh()}catch(e){$("uploadStatus").textContent=e.message}finally{working(false)}};
$("refresh").onclick=refresh;$("reset").onclick=async()=>{if(!confirm("Delete all indexed documents?"))return;try{await api("/api/documents",{method:"DELETE"});showDocs([]);$("count").textContent="0 chunks";history=[];$("uploadStatus").textContent="Knowledge base cleared."}catch(e){$("uploadStatus").textContent=e.message}};
$("form").onsubmit=async e=>{e.preventDefault();if(busy)return;let q=$("question").value.trim();if(!q)return;$("question").value="";msg("user",q);let wait=msg("assistant","Searching your documents…");working(true);try{let d=await api("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:q,history,top_k:4})});wait.querySelector(".answer").textContent=d.answer||"No answer returned.";if(d.sources?.length){let s=document.createElement("details");s.className="sources";s.innerHTML='<summary>View '+d.sources.length+' retrieved sources</summary>'+d.sources.map(x=>'<div class="source"><b>['+esc(x.id)+'] '+esc(x.source)+'</b> · chunk '+esc(x.chunk)+'<p>'+esc(x.excerpt)+'</p></div>').join("");wait.querySelector(".bubble").append(s)}history.push({role:"user",content:q},{role:"assistant",content:d.answer||""});history=history.slice(-8)}catch(e){wait.querySelector(".answer").textContent="Could not generate an answer: "+e.message}finally{working(false)}};
document.querySelectorAll("[data-q]").forEach(b=>b.onclick=()=>{$("question").value=b.dataset.q;$("question").focus()});$("login").onsubmit=async e=>{e.preventDefault();try{await api("/api/login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({password:$("password").value})});$("modal").classList.add("hidden");$("password").value="";init()}catch(e){$("loginError").textContent=e.message}};$("logout").onclick=async()=>{await api("/api/logout",{method:"POST"});$("modal").classList.remove("hidden")};init();
</script></body></html>"""
@app.get("/",response_class=HTMLResponse,include_in_schema=False)
def home(): return PAGE
@app.get("/api/health")
def health(): return {"status":"ok","service":"Atlas RAG Assistant","groq_configured":bool(os.getenv("GROQ_API_KEY","").strip()),"password_protection_enabled":bool(PASSWORD)}
@app.get("/api/session")
def session(req:Request): return {"auth_required":bool(PASSWORD),"authenticated":(not PASSWORD) or bool(req.session.get("ok"))}
@app.post("/api/login")
def login(body:Login,req:Request):
 if not PASSWORD: req.session["ok"]=True; return {"ok":True}
 if not secrets.compare_digest(body.password,PASSWORD): raise HTTPException(401,"Incorrect workspace password.")
 req.session["ok"]=True; return {"ok":True}
@app.post("/api/logout")
def logout(req:Request): req.session.clear(); return {"ok":True}
@app.get("/api/documents")
def documents(req:Request):
 auth(req)
 try:
  r=rag(); return {"documents":r.list_documents(),"total_chunks":r.collection.count()}
 except Exception as e: raise HTTPException(500,f"Could not load the index: {e}") from e
@app.post("/api/documents/index")
def index(req:Request,files:list[UploadFile]=File(...)):
 auth(req)
 if not files: raise HTTPException(400,"Choose one or more files.")
 payload=[]
 for f in files:
  name=Path(f.filename or "").name
  if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS: raise HTTPException(415,f"Unsupported file {name}; use PDF, TXT, MD, DOCX.")
  data=f.file.read(int(os.getenv("MAX_UPLOAD_MB","20"))*1024*1024+1)
  if len(data)>int(os.getenv("MAX_UPLOAD_MB","20"))*1024*1024: raise HTTPException(413,f"{name} exceeds the upload limit.")
  payload.append((name,data))
 try:
  out=rag().add_documents(payload); out["documents"]=rag().list_documents(); return out
 except Exception as e: raise HTTPException(500,f"Indexing failed: {e}") from e
@app.delete("/api/documents")
def reset(req:Request):
 auth(req)
 try: rag().reset(); return {"ok":True,"documents":[],"total_chunks":0}
 except Exception as e: raise HTTPException(500,f"Could not reset: {e}") from e
@app.post("/api/chat")
def chat(body:Chat,req:Request):
 auth(req)
 try: return rag().ask(body.message,body.top_k,body.history[-8:])
 except ValueError as e: raise HTTPException(400,str(e)) from e
 except Exception as e: raise HTTPException(502,f"Answer generation failed: {e}") from e
if __name__=="__main__":
 import uvicorn
 uvicorn.run("app:app",host="0.0.0.0",port=int(os.getenv("PORT","8000")))
