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
