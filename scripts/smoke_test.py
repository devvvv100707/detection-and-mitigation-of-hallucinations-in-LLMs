"""Check the live API response shapes. Start uvicorn before running this."""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as response:
        return response.status, json.loads(response.read().decode())


def post(path, payload):
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        return exc.code, json.loads(body) if body else {}


def main():
    status, health = get("/api/health")
    assert status == 200 and "ready" in health and "error" in health
    for _ in range(120):
        status, health = get("/api/health")
        if health["ready"] or health["error"]:
            break
        time.sleep(2)
    status, results = get("/api/results")
    assert status == 200
    for key in ("original_experiment", "reproduced_baseline", "comparison", "label_metrics"):
        assert key in results
    assert "validation" in results["comparison"] and "test" in results["comparison"]
    code, bad = post("/api/analyze", {"question": "   "})
    assert code == 422
    code, long = post("/api/analyze", {"question": "a" * 501})
    assert code == 422
    if not health["ready"]:
        code, not_ready = post("/api/analyze", {"question": "What is anemia?"})
        assert code == 503
        print("smoke test: models not ready, 503 shape checked")
        return 0
    code, analyzed = post("/api/analyze", {"question": "What is anemia?"})
    assert code == 200
    for key in (
        "student_answer", "teacher_answer", "final_answer", "tier",
        "student_verification", "teacher_verification", "similarities",
        "evidence", "reasons", "latency", "settings",
    ):
        assert key in analyzed, key
    assert analyzed["tier"] in ("accepted", "revised", "uncertain")
    assert analyzed["student_verification"] in ("Supported", "Insufficient evidence")
    name = results["figures"][0]
    with urllib.request.urlopen(BASE + "/api/figures/" + name, timeout=30) as response:
        assert response.status == 200
        assert response.headers["content-type"].startswith("image/")
    missing = urllib.request.Request(BASE + "/api/figures/../main.py")
    try:
        urllib.request.urlopen(missing, timeout=10)
        raise SystemExit("path traversal was not rejected")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
