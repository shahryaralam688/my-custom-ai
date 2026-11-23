import mimetypes
import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from middleware import db

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

AGENT_INSTRUCTIONS = """
You are an Agentic Coding Assistant with access to Cloud Workspace file tools.
When the user asks you to build a website, app, or write code files, create real files using these XML tool blocks:

<tool_call name="write_file" path="index.html">
...full file content...
</tool_call>

<tool_call name="write_file" path="styles.css">
...full file content...
</tool_call>

You can also inspect existing files using:
<tool_call name="list_files"></tool_call>
<tool_call name="read_file" path="index.html"></tool_call>

Rules:
- Always write complete, working files with no placeholders.
- Keep your explanation concise after creating the files.
"""

app = FastAPI(title="Custom Qwen3 AI — Middle Observability & Chat Hub")
app.state.db_path = db.DEFAULT_DB_PATH


@app.on_event("startup")
def on_startup():
    db.init_db(app.state.db_path)


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def parse_thinking_response(raw_text: str) -> Tuple[str, str]:
    """Separate <think>...</think> reasoning trace from the final answer."""
    match = re.search(r"<think>(.*?)</think>(.*)", raw_text, flags=re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        final = match.group(2).strip()
        return thinking, final
    return "", raw_text.strip()


def extract_tool_calls(text: str) -> Tuple[List[Dict[str, str]], str]:
    """Extract <tool_call name="..." path="...">...</tool_call> blocks and return cleaned text."""
    calls: List[Dict[str, str]] = []
    pattern = re.compile(
        r'<tool_call\s+name="([^"]+)"(?:\s+path="([^"]*)")?\s*>(.*?)</tool_call>',
        flags=re.DOTALL,
    )
    for m in pattern.finditer(text):
        calls.append({
            "name": m.group(1).strip(),
            "path": (m.group(2) or "").strip(),
            "content": m.group(3).strip(),
        })
    cleaned = pattern.sub("", text).strip()
    return calls, cleaned


class ChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] = []
    enable_thinking: bool = True
    agent_mode: bool = True


class FeedbackRequest(BaseModel):
    interaction_id: int
    feedback: int  # 1 = thumbs up, -1 = thumbs down, 0 = neutral
    corrected_response: Optional[str] = None


class SettingsRequest(BaseModel):
    colab_url: Optional[str] = None
    system_prompt: Optional[str] = None
    active_version: Optional[str] = None


class VersionRequest(BaseModel):
    version: str
    base_model: str = "Qwen/Qwen3-32B"
    train_loss: Optional[float] = None
    dataset_notes: str = ""


class EvalRequest(BaseModel):
    version: Optional[str] = None
    prompts: Optional[List[str]] = None


async def _call_colab_model(
    colab_url: str,
    prompt: str,
    system_prompt: str,
    history: List[Dict[str, str]],
    enable_thinking: bool = True,
) -> Dict[str, Any]:
    url = colab_url.rstrip("/") + "/chat"
    payload = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "history": history,
        "enable_thinking": enable_thinking,
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


@app.get("/", response_class=HTMLResponse)
def serve_chat_ui():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    dash_file = FRONTEND_DIR / "dashboard.html"
    if not dash_file.exists():
        return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)
    return HTMLResponse(dash_file.read_text(encoding="utf-8"))


@app.get("/api/health")
async def check_health():
    settings = db.get_settings(app.state.db_path)
    colab_url = settings.get("colab_url", "").strip()
    if not colab_url:
        return {
            "middleware": "ok",
            "colab_connected": False,
            "message": "Colab Tunnel URL not configured yet. Open /dashboard to paste your URL.",
        }
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(colab_url.rstrip("/") + "/health")
            r.raise_for_status()
            return {
                "middleware": "ok",
                "colab_connected": True,
                "colab_info": r.json(),
            }
    except Exception as exc:
        return {
            "middleware": "ok",
            "colab_connected": False,
            "message": f"Cannot reach Colab URL: {exc}",
        }


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    db.init_db(app.state.db_path)
    settings = db.get_settings(app.state.db_path)
    colab_url = settings.get("colab_url", "").strip()
    system_prompt = settings.get("system_prompt", db.DEFAULT_SYSTEM_PROMPT)
    if req.agent_mode:
        system_prompt = system_prompt + "\n\n" + AGENT_INSTRUCTIONS
    active_version = settings.get("active_version", "v1.0-base")

    start = time.perf_counter()
    created_files: List[str] = []

    if not colab_url:
        lower_p = req.prompt.lower()
        if req.agent_mode and any(k in lower_p for k in ["website", "web", "html", "portfolio", "landing", "site", "page"]):
            thinking = (
                "Demo Agent Mode: User ne website banane ko kaha hai. "
                "Main `index.html`, `styles.css`, aur `script.js` cloud storage mein create kar raha hoon."
            )
            db.save_generated_file(
                "index.html",
                """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Custom AI Demo Site</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <nav class="nav">
    <div class="logo">MyAI Studio</div>
    <a href="#contact" class="cta">Get Started</a>
  </nav>
  <section class="hero">
    <h1>Built by Your Custom AI Agent</h1>
    <p>Yeh website aapke Agent Mode ne cloud storage mein khud banayi hai. Right-side panel mein files aur Live Preview dekhein!</p>
    <button id="demoBtn">Test Interactive JS</button>
    <p id="demoMsg"></p>
  </section>
  <section class="cards">
    <div class="card"><h3>Fast</h3><p>Clean HTML5 &amp; CSS Grid layout.</p></div>
    <div class="card"><h3>Agentic</h3><p>Files created automatically in cloud storage.</p></div>
    <div class="card"><h3>Editable</h3><p>Colab A100 connect karke kuch bhi customize karein.</p></div>
  </section>
  <script src="script.js"></script>
</body>
</html>""",
                db_path=app.state.db_path,
            )
            db.save_generated_file(
                "styles.css",
                """* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #0b0f19; color: #f1f5f9; }
.nav { display: flex; justify-content: space-between; align-items: center; padding: 18px 40px; border-bottom: 1px solid #1e293b; }
.logo { font-weight: 700; font-size: 18px; color: #a78bfa; }
.cta { background: #7c5cff; color: #fff; text-decoration: none; padding: 8px 16px; border-radius: 8px; }
.hero { text-align: center; padding: 80px 20px 50px; max-width: 720px; margin: 0 auto; }
.hero h1 { font-size: 38px; margin-bottom: 14px; }
.hero p { color: #94a3b8; margin-bottom: 24px; line-height: 1.6; }
button { background: #7c5cff; color: #fff; border: none; padding: 12px 22px; border-radius: 10px; cursor: pointer; font-weight: 600; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; max-width: 900px; margin: 20px auto 60px; padding: 0 20px; }
.card { background: #131a2a; border: 1px solid #1e293b; padding: 20px; border-radius: 12px; }
.card h3 { margin-bottom: 8px; color: #c4b5fd; }
.card p { font-size: 14px; color: #94a3b8; }""",
                db_path=app.state.db_path,
            )
            db.save_generated_file(
                "script.js",
                """document.getElementById('demoBtn').addEventListener('click', () => {
  document.getElementById('demoMsg').textContent = 'JavaScript is working live inside Cloud Preview!';
});""",
                db_path=app.state.db_path,
            )
            created_files = ["index.html", "styles.css", "script.js"]
            final_answer = (
                "Maine **Demo Agent Mode** mein aapke liye 3 files cloud storage mein bana di hain:\n"
                "- `index.html`\n- `styles.css`\n- `script.js`\n\n"
                "Right side par **Workspace Files & Live Preview** panel mein click karke website live dekh lein! "
                "(Apne asli **Qwen3-32B** model se banwane ke liye Colab notebook run karke `/dashboard` mein URL paste karein.)"
            )
        else:
            thinking = "Demo Mode: Colab GPU URL abhi set nahi hai."
            final_answer = (
                "Abhi aap **Demo Mode** mein hain kyunki Colab Tunnel URL set nahi hai.\n\n"
                "1. Agent Mode test karne ke liye likhein: **\"Build a modern website for my AI agency\"**.\n"
                "2. Asli **Qwen3-32B** chalane ke liye Colab mein `02_serve_qwen3_colab.ipynb` run karke URL `/dashboard` par paste karein."
            )
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
    else:
        try:
            current_prompt = req.prompt
            current_history = list(req.history)
            thinking_parts: List[str] = []
            final_answer = ""

            max_turns = 4 if req.agent_mode else 1
            for _ in range(max_turns):
                data = await _call_colab_model(
                    colab_url=colab_url,
                    prompt=current_prompt,
                    system_prompt=system_prompt,
                    history=current_history,
                    enable_thinking=req.enable_thinking,
                )
                raw_text = data.get("response", "")
                t_part, f_part = parse_thinking_response(raw_text)
                if data.get("thinking"):
                    t_part = data["thinking"]
                if data.get("final_response"):
                    f_part = data["final_response"]
                if data.get("model_version"):
                    active_version = data["model_version"]
                if t_part:
                    thinking_parts.append(t_part)

                if not req.agent_mode:
                    final_answer = f_part
                    break

                calls, cleaned_text = extract_tool_calls(f_part)
                if not calls:
                    final_answer = cleaned_text
                    break

                tool_outputs: List[str] = []
                needs_followup = False
                for call in calls:
                    cname, cpath, ccontent = call["name"], call["path"], call["content"]
                    if cname == "write_file" and cpath:
                        db.save_generated_file(cpath, ccontent, db_path=app.state.db_path)
                        if cpath not in created_files:
                            created_files.append(cpath)
                        tool_outputs.append(f"Saved `{cpath}` to cloud storage.")
                    elif cname == "read_file" and cpath:
                        fobj = db.get_generated_file(cpath, db_path=app.state.db_path)
                        tool_outputs.append(
                            f"Contents of `{cpath}`:\n{fobj['content']}" if fobj else f"File `{cpath}` not found."
                        )
                        needs_followup = True
                    elif cname == "list_files":
                        flist = [f["filename"] for f in db.get_generated_files(db_path=app.state.db_path)]
                        tool_outputs.append(f"Workspace files: {flist}")
                        needs_followup = True

                if not needs_followup:
                    final_answer = cleaned_text or f"Created files: {', '.join(created_files)}"
                    break

                current_history.append({"role": "user", "content": current_prompt})
                current_history.append({"role": "assistant", "content": f_part})
                current_prompt = "Tool results:\n" + "\n".join(tool_outputs) + "\nContinue and finish the task."
                final_answer = cleaned_text

            thinking = "\n\n".join(thinking_parts)
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Colab model error: {exc}")

    interaction_id = db.log_interaction(
        prompt=req.prompt,
        thinking_trace=thinking,
        final_response=final_answer,
        model_version=active_version,
        latency_ms=latency_ms,
        db_path=app.state.db_path,
    )

    return {
        "interaction_id": interaction_id,
        "model_version": active_version,
        "thinking": thinking,
        "final_response": final_answer,
        "created_files": created_files,
        "latency_ms": latency_ms,
    }


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    ok = db.update_feedback(
        interaction_id=req.interaction_id,
        feedback=req.feedback,
        corrected_response=req.corrected_response,
        db_path=app.state.db_path,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Interaction ID not found")
    return {"status": "saved", "interaction_id": req.interaction_id}


@app.get("/api/logs")
def list_logs(limit: int = 50):
    return {
        "stats": db.get_stats(app.state.db_path),
        "logs": db.get_logs(limit=limit, db_path=app.state.db_path),
    }


@app.get("/api/settings")
def read_settings():
    return db.get_settings(app.state.db_path)


@app.post("/api/settings")
def write_settings(req: SettingsRequest):
    if req.colab_url is not None:
        db.update_setting("colab_url", req.colab_url.strip(), app.state.db_path)
    if req.system_prompt is not None:
        db.update_setting("system_prompt", req.system_prompt.strip(), app.state.db_path)
    if req.active_version is not None:
        db.update_setting("active_version", req.active_version.strip(), app.state.db_path)
    return db.get_settings(app.state.db_path)


@app.get("/api/versions")
def list_versions():
    return {"versions": db.get_versions(app.state.db_path)}


@app.post("/api/versions")
def add_version(req: VersionRequest):
    db.register_version(
        version=req.version,
        base_model=req.base_model,
        train_loss=req.train_loss,
        dataset_notes=req.dataset_notes,
        db_path=app.state.db_path,
    )
    db.update_setting("active_version", req.version, app.state.db_path)
    return {"status": "registered", "version": req.version}


@app.post("/api/run-eval")
async def run_eval(req: EvalRequest):
    db.init_db(app.state.db_path)
    settings = db.get_settings(app.state.db_path)
    colab_url = settings.get("colab_url", "").strip()
    if not colab_url:
        raise HTTPException(status_code=400, detail="Set Colab Tunnel URL first before running evaluation.")

    version = req.version or settings.get("active_version", "v1.0-base")
    system_prompt = settings.get("system_prompt", db.DEFAULT_SYSTEM_PROMPT)
    prompts = req.prompts or db.DEFAULT_EVAL_PROMPTS

    results = []
    for idx, prompt in enumerate(prompts):
        start = time.perf_counter()
        data = await _call_colab_model(
            colab_url=colab_url,
            prompt=prompt,
            system_prompt=system_prompt,
            history=[],
            enable_thinking=True,
        )
        raw_text = data.get("response", "")
        thinking, final_answer = parse_thinking_response(raw_text)
        if data.get("thinking"):
            thinking = data["thinking"]
        if data.get("final_response"):
            final_answer = data["final_response"]
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        db.save_eval_snapshot(
            version=version,
            prompt_index=idx,
            prompt=prompt,
            thinking_trace=thinking,
            final_response=final_answer,
            latency_ms=latency_ms,
            db_path=app.state.db_path,
        )
        results.append({
            "prompt_index": idx,
            "prompt": prompt,
            "thinking": thinking,
            "final_response": final_answer,
            "latency_ms": latency_ms,
        })

    return {"version": version, "results": results}


@app.get("/api/eval-compare")
def compare_evals(v1: Optional[str] = None, v2: Optional[str] = None):
    snapshots = db.get_eval_snapshots(db_path=app.state.db_path)
    available_versions = sorted(list({s["version"] for s in snapshots}))
    if not v1 and len(available_versions) >= 1:
        v1 = available_versions[0]
    if not v2 and len(available_versions) >= 2:
        v2 = available_versions[-1]

    v1_map = {}
    v2_map = {}
    for s in snapshots:
        if s["version"] == v1 and s["prompt_index"] not in v1_map:
            v1_map[s["prompt_index"]] = s
        if s["version"] == v2 and s["prompt_index"] not in v2_map:
            v2_map[s["prompt_index"]] = s

    all_indices = sorted(list(set(v1_map.keys()) | set(v2_map.keys())))
    comparisons = []
    for idx in all_indices:
        comparisons.append({
            "prompt_index": idx,
            "prompt": (v1_map.get(idx) or v2_map.get(idx))["prompt"],
            "v1": v1_map.get(idx),
            "v2": v2_map.get(idx),
        })

    return {
        "available_versions": available_versions,
        "v1": v1,
        "v2": v2,
        "comparisons": comparisons,
    }


@app.get("/api/export-dataset")
def export_dataset():
    jsonl_content = db.export_training_jsonl(app.state.db_path)
    return PlainTextResponse(
        content=jsonl_content,
        media_type="application/jsonl",
        headers={"Content-Disposition": "attachment; filename=training_feedback.jsonl"},
    )


@app.get("/api/files")
def list_workspace_files():
    return {"files": db.get_generated_files(db_path=app.state.db_path)}


@app.delete("/api/files")
def reset_workspace_files():
    db.clear_generated_files(db_path=app.state.db_path)
    return {"status": "cleared"}


@app.get("/api/preview/{filename:path}")
def preview_workspace_file(filename: str):
    fobj = db.get_generated_file(filename, db_path=app.state.db_path)
    if not fobj:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in cloud workspace.")
    mime_type, _ = mimetypes.guess_type(filename)
    return Response(content=fobj["content"], media_type=mime_type or "text/plain; charset=utf-8")
