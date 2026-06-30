from openpi.training import prompt_bakeoff as pb


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
