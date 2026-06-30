import json
import pathlib

from openpi.training import prompt_bakeoff as pb
from openpi.training.memory_labels import Episode, MemoryLabelConfig, MemoryLabelGenerator, MemoryLabels


def _good():
    return dict(faithfulness=0.9, conciseness=0.7, decision_relevance=0.9,
                temporal_coherence=0.9, determinism=0.8, structural=1.0)


def test_composite_rewards_good_prompt():
    assert pb.composite_score(_good()) is not None
    assert pb.composite_score(_good()) > 0.7


def test_gate_rejects_short_but_unfaithful():
    m = _good()
    m["faithfulness"] = 0.1  # below gate
    m["conciseness"] = 1.0   # maximally short
    assert pb.composite_score(m) is None


def test_gate_rejects_incoherent():
    m = _good()
    m["temporal_coherence"] = 0.2
    assert pb.composite_score(m) is None


def test_rank_puts_gated_out_last():
    good, bad = _good(), _good()
    bad["decision_relevance"] = 0.1
    ranked = pb.rank_candidates({"good": good, "bad": bad})
    assert ranked[0][0] == "good"
    assert ranked[-1][0] == "bad" and ranked[-1][1] is None


def test_score_labels_ranges():
    eps = [Episode(goal="g", subtasks=["pick cup", "place cup"], success_flags=[True, True])]
    labels = [MemoryLabels(episode_id="0", memories=["picked cup", "placed cup"])]
    s = pb.score_labels(eps, labels)
    assert 0.0 <= s["faithfulness"] <= 1.0
    assert 0.0 <= s["conciseness"] <= 1.0
    assert 0.0 <= s["structural"] <= 1.0


def test_load_prompts(tmp_path):
    (tmp_path / "a.txt").write_text("A {goal} {previous_memory} {new_event}")
    (tmp_path / "b.txt").write_text("B")
    got = pb.load_prompts(tmp_path)
    assert set(got) == {"a", "b"} and got["a"].startswith("A ")


def test_run_bakeoff_mock_ranks_and_writes_shards(tmp_path):
    eps = [Episode(goal="g", subtasks=["a", "b", "c"], success_flags=[True, True, True])]
    pdir = tmp_path / "prompts"
    pdir.mkdir()
    (pdir / "p1.txt").write_text("{goal}|{previous_memory}|{new_event}")
    out = tmp_path / "out"

    def factory(template):
        return MemoryLabelConfig(backend="mock", generation_mode="recursive", prompt_template=template)

    result = pb.run_bakeoff(eps, pdir, factory, out_dir=out)
    assert "p1" in result["metrics"]
    assert (out / "p1" / "0.json").exists()  # durable shard written
    assert result["ranking"][0][0] == "p1"
