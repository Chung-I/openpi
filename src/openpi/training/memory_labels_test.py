import asyncio
import json

from openpi.training.memory_labels import Episode
from openpi.training.memory_labels import MemoryLabelConfig
from openpi.training.memory_labels import MemoryLabelGenerator
from openpi.training.memory_labels import MemoryLabels


def test_mock_backend_generates_labels():
    config = MemoryLabelConfig(backend="mock")
    generator = MemoryLabelGenerator(config)

    episodes = [
        Episode(
            goal="clean the kitchen",
            subtasks=[
                "pick up plate",
                "place plate in cabinet",
                "wipe counter",
                "wipe counter",
            ],
            success_flags=[True, True, False, True],
        )
    ]

    labels = generator.generate_labels(episodes)
    assert len(labels) == 1
    result = labels[0]
    assert isinstance(result, MemoryLabels)
    assert result.episode_id == "0"
    # Should have one memory string per subtask
    assert len(result.memories) == 4
    # Each label should be a string
    assert all(isinstance(m, str) for m in result.memories)


def test_config_defaults():
    config = MemoryLabelConfig()
    assert config.backend == "claude"
    assert config.max_memory_tokens == 128


def test_resolved_api_key_env_defaults():
    claude_config = MemoryLabelConfig(backend="claude")
    assert claude_config.resolved_api_key_env == "ANTHROPIC_API_KEY"

    openai_config = MemoryLabelConfig(backend="openai")
    assert openai_config.resolved_api_key_env == "OPENAI_API_KEY"

    explicit_config = MemoryLabelConfig(backend="openai", api_key_env="MY_CUSTOM_KEY")
    assert explicit_config.resolved_api_key_env == "MY_CUSTOM_KEY"


def test_openai_client_uses_base_url(monkeypatch):
    import pytest

    openai = pytest.importorskip("openai")
    captured = {}

    class _FakeOpenAI:
        def __init__(self, *, base_url=None, api_key=None):
            captured["base_url"] = base_url
            captured["api_key"] = api_key

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = MemoryLabelConfig(backend="openai", base_url="http://localhost:8000/v1", model="qwen")
    gen = MemoryLabelGenerator(config)
    gen._get_client()
    assert captured["base_url"] == "http://localhost:8000/v1"
    assert captured["api_key"] == "EMPTY"  # vLLM ignores key; fall back when env unset


def test_config_has_concurrency_default():
    assert MemoryLabelConfig().max_concurrency == 64


def test_generate_labels_async_matches_mock():
    config = MemoryLabelConfig(backend="mock", max_concurrency=4)
    gen = MemoryLabelGenerator(config)
    eps = [Episode(goal="g", subtasks=["a", "b", "c"], success_flags=[True, True, True])]
    labels = asyncio.run(gen.generate_labels_async(eps))
    assert len(labels) == 1
    assert len(labels[0].memories) == 3
    assert all(isinstance(m, str) for m in labels[0].memories)


def test_generate_labels_async_respects_concurrency(monkeypatch):
    config = MemoryLabelConfig(backend="openai", base_url="http://x/v1", model="qwen", max_concurrency=2)
    gen = MemoryLabelGenerator(config)

    state = {"in_flight": 0, "max": 0}

    class _Msg:
        def __init__(self, c):
            self.message = type("M", (), {"content": c})

    class _Resp:
        def __init__(self, c):
            self.choices = [_Msg(c)]

    class _Completions:
        async def create(self, **kw):
            state["in_flight"] += 1
            state["max"] = max(state["max"], state["in_flight"])
            await asyncio.sleep(0.01)
            state["in_flight"] -= 1
            return _Resp("mem")

    class _Chat:
        completions = _Completions()

    class _FakeAsync:
        chat = _Chat()

    monkeypatch.setattr(gen, "_get_async_client", lambda: _FakeAsync())
    eps = [Episode(goal="g", subtasks=[str(i) for i in range(8)], success_flags=[True] * 8)]
    asyncio.run(gen.generate_labels_async(eps))
    assert state["max"] <= 2  # never exceeds max_concurrency


def test_generate_labels_async_resumes(tmp_path):
    config = MemoryLabelConfig(backend="mock")
    gen = MemoryLabelGenerator(config)
    eps = [
        Episode(goal="g0", subtasks=["a"], success_flags=[True]),
        Episode(goal="g1", subtasks=["b"], success_flags=[True]),
    ]
    # Pre-seed episode 0's shard with a sentinel; it must NOT be regenerated.
    (tmp_path / "0.json").write_text(json.dumps({"episode_id": "0", "memories": ["SENTINEL"]}))
    labels = asyncio.run(gen.generate_labels_async(eps, out_dir=tmp_path))
    assert labels[0].memories == ["SENTINEL"]  # resumed, not overwritten
    assert (tmp_path / "1.json").exists()  # episode 1 generated
    assert labels[1].memories[0].startswith("Memory summary")


def test_generate_labels_async_full_resume_no_client(tmp_path):
    # All episodes already checkpointed -> must NOT construct a client (which would
    # raise for the 'local' backend), just return the resumed labels.
    config = MemoryLabelConfig(backend="local")  # _get_async_client would raise NotImplementedError
    gen = MemoryLabelGenerator(config)
    eps = [Episode(goal="g0", subtasks=["a"], success_flags=[True])]
    (tmp_path / "0.json").write_text(json.dumps({"episode_id": "0", "memories": ["DONE"]}))
    labels = asyncio.run(gen.generate_labels_async(eps, out_dir=tmp_path))
    assert labels[0].memories == ["DONE"]


def _fake_openai_client(captured):
    class _Msg:
        def __init__(self, c):
            self.message = type("M", (), {"content": c})

    class _Resp:
        def __init__(self, c):
            self.choices = [_Msg(c)]

    class _Completions:
        def create(self, **kw):
            captured.update(kw)
            return _Resp("mem")

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    return _Client()


def test_disable_thinking_passes_extra_body():
    captured = {}
    gen = MemoryLabelGenerator(MemoryLabelConfig(backend="openai", model="qwen", disable_thinking=True))
    gen._client = _fake_openai_client(captured)  # noqa: SLF001 -- bypass real client construction
    gen._generate_single("g", "1. [SUCCESS] x")  # noqa: SLF001
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_thinking_enabled_by_default_sends_no_extra_body():
    captured = {}
    gen = MemoryLabelGenerator(MemoryLabelConfig(backend="openai", model="qwen"))
    gen._client = _fake_openai_client(captured)  # noqa: SLF001
    gen._generate_single("g", "1. [SUCCESS] x")  # noqa: SLF001
    assert "extra_body" not in captured


def test_generate_labels_async_writes_shard_per_episode(tmp_path):
    config = MemoryLabelConfig(backend="mock", max_concurrency=4)
    gen = MemoryLabelGenerator(config)
    eps = [
        Episode(goal="g0", subtasks=["a", "b"], success_flags=[True, True]),
        Episode(goal="g1", subtasks=["c"], success_flags=[True]),
    ]
    asyncio.run(gen.generate_labels_async(eps, out_dir=tmp_path))
    # Each episode got its own shard written (durable resume granularity).
    assert (tmp_path / "0.json").exists()
    assert (tmp_path / "1.json").exists()
    import json as _json

    assert len(_json.loads((tmp_path / "1.json").read_text())["memories"]) == 1
