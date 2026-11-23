import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from middleware import db
from middleware.app import app, parse_thinking_response, extract_tool_calls


@pytest.fixture
def client(tmp_path: Path):
    test_db = tmp_path / "test_obs.db"
    app.state.db_path = test_db
    db.init_db(test_db)
    with TestClient(app) as c:
        yield c


def test_parse_thinking_response():
    raw = "<think>Step 1: check logic.\nStep 2: verify.</think>\nHere is the final answer."
    thinking, final = parse_thinking_response(raw)
    assert "Step 1: check logic." in thinking
    assert final == "Here is the final answer."

    t2, f2 = parse_thinking_response("Plain response without tags.")
    assert t2 == ""
    assert f2 == "Plain response without tags."


def test_frontend_routes_serve_html(client: TestClient):
    r1 = client.get("/")
    assert r1.status_code == 200
    assert "My Custom AI" in r1.text

    r2 = client.get("/dashboard")
    assert r2.status_code == 200
    assert "Before vs After" in r2.text


def test_chat_demo_mode_and_feedback_export(client: TestClient):
    res = client.post("/api/chat", json={"prompt": "Hello AI", "enable_thinking": True})
    assert res.status_code == 200
    data = res.json()
    assert data["interaction_id"] >= 1
    assert "Demo Mode" in data["final_response"]

    # Submit correction feedback
    fb_res = client.post(
        "/api/feedback",
        json={
            "interaction_id": data["interaction_id"],
            "feedback": -1,
            "corrected_response": "Assalam-o-Alaikum! Main aapki kya madad kar sakta hoon?",
        },
    )
    assert fb_res.status_code == 200

    # Export JSONL
    exp_res = client.get("/api/export-dataset")
    assert exp_res.status_code == 200
    lines = [line for line in exp_res.text.strip().splitlines() if line]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["messages"][0]["content"] == "Hello AI"
    assert "madad kar sakta hoon" in record["messages"][1]["content"]


def test_mocked_colab_eval_and_before_after_compare(client: TestClient):
    client.post("/api/settings", json={"colab_url": "https://mock-colab.trycloudflare.com"})

    # Run v1.0-base eval
    with patch("middleware.app._call_colab_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {
            "response": "<think>Basic reasoning v1</think>Answer v1",
            "model_version": "v1.0-base",
        }
        r1 = client.post("/api/run-eval", json={"version": "v1.0-base", "prompts": ["Test Q1"]})
        assert r1.status_code == 200

    # Register v1.1-qlora and run v1.1 eval
    client.post(
        "/api/versions",
        json={"version": "v1.1-qlora", "train_loss": 0.38, "dataset_notes": "Trained on 1000 samples"},
    )
    with patch("middleware.app._call_colab_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {
            "response": "<think>Deeper structured reasoning v1.1</think>Much better answer v1.1",
            "model_version": "v1.1-qlora",
        }
        r2 = client.post("/api/run-eval", json={"version": "v1.1-qlora", "prompts": ["Test Q1"]})
        assert r2.status_code == 200

    # Compare v1.0-base vs v1.1-qlora
    comp = client.get("/api/eval-compare?v1=v1.0-base&v2=v1.1-qlora")
    assert comp.status_code == 200
    cdata = comp.json()
    assert len(cdata["comparisons"]) == 1
    assert cdata["comparisons"][0]["v1"]["thinking_trace"] == "Basic reasoning v1"
    assert cdata["comparisons"][0]["v2"]["thinking_trace"] == "Deeper structured reasoning v1.1"


def test_vercel_entrypoint_imports():
    from api.index import app as vercel_app
    assert vercel_app is app



def test_agent_mode_demo_and_colab_tool_calls(client: TestClient):
    raw = '<tool_call name="write_file" path="hello.html"><h1>Hi</h1></tool_call>\nDone!'
    calls, cleaned = extract_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0]["name"] == "write_file"
    assert calls[0]["path"] == "hello.html"
    assert cleaned == "Done!"

    # Demo Agent Mode website build
    res = client.post("/api/chat", json={"prompt": "Build a modern website", "agent_mode": True})
    assert res.status_code == 200
    data = res.json()
    assert "index.html" in data["created_files"]

    files_res = client.get("/api/files")
    assert files_res.status_code == 200
    filenames = [f["filename"] for f in files_res.json()["files"]]
    assert "index.html" in filenames and "styles.css" in filenames

    prev = client.get("/api/preview/index.html")
    assert prev.status_code == 200
    assert "text/html" in prev.headers["content-type"]
    assert "Built by Your Custom AI Agent" in prev.text
