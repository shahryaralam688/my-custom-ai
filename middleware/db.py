import os
import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

DEFAULT_DB_PATH = (
    Path("/tmp/ai_observability.db")
    if os.environ.get("VERCEL")
    else Path(__file__).resolve().parent.parent / "data" / "ai_observability.db"
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, accurate, and thoughtful AI assistant. "
    "Think step-by-step carefully before answering. "
    "Provide clear, well-structured, concise responses with clean formatting."
)

DEFAULT_EVAL_PROMPTS = [
    "Write a Python function to check if a string is a palindrome, ignoring spaces and case, and explain its time complexity.",
    "Mujhe samjhao ke Transformer model mein Self-Attention kya kaam karta hai, simple aur clear example ke sath.",
    "A farmer has 17 sheep. All but 9 run away. How many sheep are left, and why?",
]


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_version TEXT NOT NULL,
            prompt TEXT NOT NULL,
            thinking_trace TEXT DEFAULT '',
            final_response TEXT NOT NULL,
            latency_ms REAL DEFAULT 0,
            feedback INTEGER DEFAULT 0,
            corrected_response TEXT DEFAULT ''
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT UNIQUE NOT NULL,
            base_model TEXT NOT NULL,
            train_loss REAL,
            dataset_notes TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS eval_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT NOT NULL,
            prompt_index INTEGER NOT NULL,
            prompt TEXT NOT NULL,
            thinking_trace TEXT DEFAULT '',
            final_response TEXT NOT NULL,
            latency_ms REAL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS generated_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL DEFAULT 'default',
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(session_id, filename)
        )
    """)

    # Seed default settings if missing
    cur.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('colab_url', '')")
    cur.execute(
        "INSERT OR IGNORE INTO settings (key, value) VALUES ('system_prompt', ?)",
        (DEFAULT_SYSTEM_PROMPT,),
    )
    cur.execute(
        "INSERT OR IGNORE INTO settings (key, value) VALUES ('active_version', 'v1.0-base')"
    )

    # Seed initial base version if empty
    cur.execute(
        """
        INSERT OR IGNORE INTO model_versions (version, base_model, train_loss, dataset_notes, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "v1.0-base",
            "Qwen/Qwen3-32B",
            None,
            "Initial base model before custom fine-tuning",
            datetime.now(timezone.utc).isoformat(),
        ),
    )

    conn.commit()
    conn.close()


def get_settings(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, str]:
    conn = get_connection(db_path)
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    conn.close()
    return {row["key"]: row["value"] for row in rows}


def update_setting(key: str, value: str, db_path: Path = DEFAULT_DB_PATH) -> None:
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()
    conn.close()


def log_interaction(
    prompt: str,
    thinking_trace: str,
    final_response: str,
    model_version: str,
    latency_ms: float,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO interactions (timestamp, model_version, prompt, thinking_trace, final_response, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            model_version,
            prompt,
            thinking_trace,
            final_response,
            latency_ms,
        ),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(row_id)


def update_feedback(
    interaction_id: int,
    feedback: int,
    corrected_response: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> bool:
    conn = get_connection(db_path)
    cur = conn.cursor()
    if corrected_response is not None:
        cur.execute(
            "UPDATE interactions SET feedback = ?, corrected_response = ? WHERE id = ?",
            (feedback, corrected_response, interaction_id),
        )
    else:
        cur.execute(
            "UPDATE interactions SET feedback = ? WHERE id = ?",
            (feedback, interaction_id),
        )
    updated = cur.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def get_logs(limit: int = 50, db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM interactions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, Any]:
    conn = get_connection(db_path)
    row = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN feedback = -1 THEN 1 ELSE 0 END) as negative,
            SUM(CASE WHEN corrected_response != '' THEN 1 ELSE 0 END) as corrected,
            AVG(latency_ms) as avg_latency_ms
        FROM interactions
    """).fetchone()
    conn.close()
    return {
        "total": row["total"] or 0,
        "positive": row["positive"] or 0,
        "negative": row["negative"] or 0,
        "corrected": row["corrected"] or 0,
        "avg_latency_ms": round(row["avg_latency_ms"] or 0.0, 1),
    }


def register_version(
    version: str,
    base_model: str,
    train_loss: Optional[float],
    dataset_notes: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    conn = get_connection(db_path)
    conn.execute(
        """
        INSERT INTO model_versions (version, base_model, train_loss, dataset_notes, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(version) DO UPDATE SET
            base_model=excluded.base_model,
            train_loss=excluded.train_loss,
            dataset_notes=excluded.dataset_notes
        """,
        (
            version,
            base_model,
            train_loss,
            dataset_notes,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_versions(db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    rows = conn.execute("SELECT * FROM model_versions ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_eval_snapshot(
    version: str,
    prompt_index: int,
    prompt: str,
    thinking_trace: str,
    final_response: str,
    latency_ms: float,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    conn = get_connection(db_path)
    conn.execute(
        """
        INSERT INTO eval_snapshots (version, prompt_index, prompt, thinking_trace, final_response, latency_ms, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            version,
            prompt_index,
            prompt,
            thinking_trace,
            final_response,
            latency_ms,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_eval_snapshots(version: Optional[str] = None, db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    if version:
        rows = conn.execute(
            "SELECT * FROM eval_snapshots WHERE version = ? ORDER BY prompt_index ASC, id DESC",
            (version,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM eval_snapshots ORDER BY id DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def export_training_jsonl(db_path: Path = DEFAULT_DB_PATH) -> str:
    """Export corrected or positively rated interactions as SFT-ready JSONL."""
    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT prompt, final_response, corrected_response, feedback
        FROM interactions
        WHERE corrected_response != '' OR feedback = 1
        ORDER BY id ASC
    """).fetchall()
    conn.close()

    lines = []
    for row in rows:
        target = row["corrected_response"].strip() if row["corrected_response"].strip() else row["final_response"].strip()
        record = {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": target},
            ]
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines)


def save_generated_file(
    filename: str,
    content: str,
    session_id: str = "default",
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    clean_name = filename.lstrip("/").replace("..", "")
    conn = get_connection(db_path)
    conn.execute(
        """
        INSERT INTO generated_files (session_id, filename, content, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session_id, filename) DO UPDATE SET
            content=excluded.content,
            updated_at=excluded.updated_at
        """,
        (session_id, clean_name, content, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_generated_files(
    session_id: str = "default",
    db_path: Path = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT filename, content, updated_at FROM generated_files WHERE session_id = ? ORDER BY filename ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_generated_file(
    filename: str,
    session_id: str = "default",
    db_path: Path = DEFAULT_DB_PATH,
) -> Optional[Dict[str, Any]]:
    clean_name = filename.lstrip("/").replace("..", "")
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT filename, content, updated_at FROM generated_files WHERE session_id = ? AND filename = ?",
        (session_id, clean_name),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def clear_generated_files(
    session_id: str = "default",
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    conn = get_connection(db_path)
    conn.execute("DELETE FROM generated_files WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
