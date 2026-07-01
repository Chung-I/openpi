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
    assert "clean kitchen" in jp
    assert "wiped once" in jp
    assert "1. wipe" in jp
    rp = mv.build_reconstruction_prompt(goal="clean kitchen", memory="wiped once")
    assert "clean kitchen" in rp
    assert "wiped once" in rp


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


def test_faithfulness_matches_stemmed_variants():
    # past-tense/plural memory forms should count as faithful vs imperative history (nltk Porter),
    # not be penalized as hallucinations: placed<->place, bowls<->bowl.
    assert mv.faithfulness("I placed three bowls in the cabinet", "place the bowl in the cabinet") >= 0.5


def test_failure_invariance_rewards_unchanged_memory():
    memories = ["I picked the apple", "I picked the apple", "I placed the apple"]
    flags = [True, False, True]  # step 1 is a failed attempt; memory stayed identical
    assert mv.failure_invariance_scores(memories, flags) == [1.0]


def test_failure_invariance_penalizes_changed_memory():
    memories = ["I picked the apple", "I failed and am retrying the apple", "I placed the apple"]
    flags = [True, False, True]
    assert mv.failure_invariance_scores(memories, flags)[0] < 1.0


def test_failure_invariance_empty_when_all_success():
    assert mv.failure_invariance_scores(["a", "b"], [True, True]) == []
