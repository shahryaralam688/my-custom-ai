import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from middleware import db

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

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


class ChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] = []
    enable_thinking: bool = True


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
    active_version = settings.get("active_version", "v1.0-base")

    start = time.perf_counter()

    if not colab_url:
        thinking = "Demo Mode: Colab GPU URL abhi set nahi hai."
        final_answer = (
            "Abhi aap **Demo Mode** mein hain kyunki Colab Tunnel URL set nahi hai.\n\n"
            "1. Google Colab mein `02_serve_qwen3_colab.ipynb` run karein.\n"
            "2. Jo `https://....trycloudflare.com` URL mile, usay `/dashboard` par paste karein.\n"
            "3. Uske baad yahan aapka apna **Qwen3-32B** model live jawab dega!"
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
    else:
        try:
            data = await _call_colab_model(
                colab_url=colab_url,
                prompt=req.prompt,
                system_prompt=system_prompt,
                history=req.history,
                enable_thinking=req.enable_thinking,
            )
            raw_text = data.get("response", "")
            thinking, final_answer = parse_thinking_response(raw_text)
            if data.get("thinking"):
                thinking = data["thinking"]
            if data.get("final_response"):
                final_answer = data["final_response"]
            if data.get("model_version"):
                active_version = data["model_version"]
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
