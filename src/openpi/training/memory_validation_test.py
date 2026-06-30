from openpi.training import memory_validation as mv


def test_compression_ratio():
    assert mv.compression_ratio("short", "a much longer cumulative history string") < 1.0
    assert mv.compression_ratio("", "abc") == 0.0


def test_faithfulness_flags_hallucination():
    history = "placed a plate in the cabinet and wiped the counter"
    faithful = mv.faithfulness("placed plate in cabinet", history)
    halluc = mv.faithfulness("launched a rocket to mars", history)
    assert faithful > 0.8
    assert halluc < 0.5


def test_faithfulness_empty_memory_is_one():
    assert mv.faithfulness("", "anything here") == 1.0


def test_prompt_builders_include_inputs():
    jp = mv.build_judge_prompt(goal="clean kitchen", history="1. wipe", memory="wiped once")
    assert "clean kitchen" in jp and "wiped once" in jp and "1. wipe" in jp
    rp = mv.build_reconstruction_prompt(goal="clean kitchen", memory="wiped once")
    assert "clean kitchen" in rp and "wiped once" in rp


def test_structural_score_clean_is_high():
    assert mv.structural_score("Placed eggplant and corn into the plate.") == 1.0


def test_structural_score_penalizes_preamble_and_markdown():
    # "here's" + "thinking"/"analyze" + "**" -> 3 of 6 checks fail -> 0.5
    assert mv.structural_score("Here's a thinking process:\n\n1. **Analyze**") <= 0.5
    assert mv.structural_score('{"memory": "x"}') < 1.0


def test_determinism_identical_samples_is_one():
    assert mv.determinism_score(["placed eggplant", "placed eggplant", "placed eggplant"]) == 1.0


def test_determinism_divergent_samples_is_low():
    assert mv.determinism_score(["placed eggplant", "opened the drawer"]) < 0.5


def test_determinism_single_sample_is_one():
    assert mv.determinism_score(["x"]) == 1.0


def test_build_coherence_prompt_mentions_both_memories():
    p = mv.build_coherence_prompt("stack bowls", "1 bowl placed", "2 bowls placed")
    assert "1 bowl placed" in p
    assert "2 bowls placed" in p
    assert "coherence" in p
