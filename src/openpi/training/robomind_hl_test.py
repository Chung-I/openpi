import json

import numpy as np
from PIL import Image

from openpi.training import robomind_hl as hl


class _FakeTok:
    """Deterministic offline tokenizer: bytes of the text (never emits 0 or 1)."""

    def encode(self, text: str) -> list[int]:
        return [b + 2 for b in text.encode("utf-8")]


def _write_frame(path):
    Image.fromarray(np.full((224, 224, 3), 128, np.uint8)).save(path)


def _row(img_rel):
    return {
        "goal": "closing a trash bin",
        "input_memory": "(none yet)",
        "target_subtask": "move towards the lid",
        "target_memory": "I moved towards the lid.",
        "image": img_rel,
    }


def test_encode_pads_and_masks():
    tok = _FakeTok()
    t, m = hl._encode(tok, "abc", 8)  # noqa: SLF001
    assert t.shape == (8,)
    assert m.shape == (8,)
    assert m.tolist() == [True, True, True, False, False, False, False, False]
    assert t[3:].tolist() == [0, 0, 0, 0, 0]  # right-padded with 0


def test_encode_empty_is_all_false():
    t, m = hl._encode(_FakeTok(), "", 5)  # noqa: SLF001
    assert not m.any()
    assert not t.any()


def test_encode_add_eos_and_truncation():
    t, m = hl._encode(_FakeTok(), "abcd", 3, add_eos=True)  # noqa: SLF001 -- 4 chars + eos, truncated to 3
    assert m.all()  # all 3 slots used
    _ = t
    t2, m2 = hl._encode(_FakeTok(), "ab", 8, add_eos=True)  # noqa: SLF001
    assert m2.tolist()[:3] == [True, True, True]
    assert t2[2] == hl._EOS_TOKEN_ID  # noqa: SLF001 -- EOS after the 2 content tokens


def test_build_hl_example(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_frame(frames / "f0.jpg")
    ex = hl.build_hl_example(_row("frames/f0.jpg"), tmp_path, _FakeTok(),
                             state_dim=32, max_prompt_tokens=16, max_memory_tokens=32, max_target_tokens=128)
    assert ex["image"].shape == (224, 224, 3)
    assert ex["image"].dtype == np.uint8
    assert ex["state"].shape == (32,)
    assert ex["tokenized_prompt"].shape == (16,)
    assert ex["tokenized_memory"].shape == (32,)
    assert ex["target_tokens"].shape == (128,)
    # target is the "Subtask: .. Memory: .." NL format, EOS-terminated
    tok = _FakeTok()
    expected = hl.HL_TARGET_TEMPLATE.format(subtask="move towards the lid", memory="I moved towards the lid.")
    exp_ids = [*tok.encode(expected), hl._EOS_TOKEN_ID]  # noqa: SLF001
    n = int(ex["target_mask"].sum())
    assert ex["target_tokens"][:n].tolist() == exp_ids
    assert ex["target_tokens"][n - 1] == hl._EOS_TOKEN_ID  # noqa: SLF001


def test_dataset_and_collate_to_observation(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(2):
        _write_frame(frames / f"f{i}.jpg")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(_row(f"frames/f{i}.jpg")) for i in range(2)) + "\n")

    ds = hl.RobomindHLDataset(manifest, tmp_path, _FakeTok(), max_prompt_tokens=16, max_memory_tokens=32, max_target_tokens=64)
    assert len(ds) == 2
    obs_dict, target_tokens, target_mask = hl.collate_hl([ds[0], ds[1]])

    assert obs_dict["image"][hl.HL_CAMERA].shape == (2, 224, 224, 3)
    assert obs_dict["image_mask"][hl.HL_CAMERA].tolist() == [True, True]
    assert obs_dict["state"].shape == (2, 32)
    assert target_tokens.shape == (2, 64)
    assert target_mask.shape == (2, 64)

    # feeds Observation.from_dict, which normalizes the uint8 frame to [-1, 1]
    from openpi.models import model as _model
    obs = _model.Observation.from_dict(obs_dict)
    img = np.asarray(obs.images[hl.HL_CAMERA])
    assert img.dtype == np.float32
    assert float(img.min()) >= -1.0
    assert float(img.max()) <= 1.0
    assert obs.tokenized_memory is not None
    assert obs.tokenized_prompt is not None


def test_question_prompt_wraps_goal(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_frame(frames / "f0.jpg")
    tok = _FakeTok()
    row = _row("frames/f0.jpg")  # goal = "closing a trash bin"
    plain = hl.build_hl_example(row, tmp_path, tok, max_prompt_tokens=32)
    ques = hl.build_hl_example(row, tmp_path, tok, max_prompt_tokens=32, question_prompt=True)
    n_plain = int(plain["tokenized_prompt_mask"].sum())
    n_ques = int(ques["tokenized_prompt_mask"].sum())
    expected = tok.encode(hl.HL_QUESTION_TEMPLATE.format(goal=row["goal"]))
    assert ques["tokenized_prompt"][:n_ques].tolist() == expected[:n_ques]
    assert n_ques > n_plain  # question wrapper adds tokens


def test_dataset_episode_filter(tmp_path):
    rows = [
        {"episode_id": "h5/241021_taskA_0/s/train/x", **_row("frames/a.jpg")},
        {"episode_id": "h5/241021_taskB_0/s/train/x", **_row("frames/b.jpg")},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    ds_all = hl.RobomindHLDataset(manifest, tmp_path, _FakeTok())
    assert len(ds_all) == 2

    ds_a = hl.RobomindHLDataset(
        manifest, tmp_path, _FakeTok(), episode_ids={"h5/241021_taskA_0/s/train/x"}
    )
    assert len(ds_a) == 1
    assert ds_a.rows[0]["episode_id"] == "h5/241021_taskA_0/s/train/x"
