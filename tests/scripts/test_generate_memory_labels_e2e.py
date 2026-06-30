import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_generate_memory_labels_mock_e2e(tmp_path):
    episodes = [{"goal": "g", "subtasks": ["a", "b"], "success_flags": [True, True]}]
    ep_file = tmp_path / "episodes.json"
    ep_file.write_text(json.dumps(episodes))
    out = tmp_path / "labels.json"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_memory_labels.py"),
         "--episodes_file", str(ep_file), "--backend", "mock", "--output", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    labels = json.loads(out.read_text())
    assert len(labels) == 1
    assert len(labels[0]["memories"]) == 2
