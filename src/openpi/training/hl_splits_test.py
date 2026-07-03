import json

from openpi.training import hl_splits


def _episode_ids():
    # 6 tasks x 5 episodes = 30 episodes; task encoded as <date>_<task>_<run>.
    ids = []
    for task in ["close_trash", "open_drawer", "pick_pear", "place_bread", "cap_lid", "in_block"]:
        for run in range(5):
            ids.append(f"h5/241021_{task}_{run}/success_episodes/train/x_{run}")
    return ids


def test_canonical_task_strips_date_and_run():
    assert hl_splits.canonical_task("h5/241021_close_trash_1/success/train/x") == "close_trash"
    assert hl_splits.canonical_task("h5/241022_open_drawer/success/train/x") == "open_drawer"


def test_build_splits_no_leakage_and_deterministic():
    ids = _episode_ids()
    s1 = hl_splits.build_splits(ids, seed=0, n_dev_unseen_tasks=2, n_dev_seen_episodes=3)
    s2 = hl_splits.build_splits(ids, seed=0, n_dev_unseen_tasks=2, n_dev_seen_episodes=3)
    assert s1 == s2  # determinism under fixed seed

    train = set(s1["train_episodes"])
    dev_seen = set(s1["dev_seen_episodes"])
    dev_unseen_tasks = set(s1["dev_unseen_tasks"])

    # (a) no episode appears in two slices
    assert train.isdisjoint(dev_seen)
    dev_unseen_eps = {e for e in ids if hl_splits.canonical_task(e) in dev_unseen_tasks}
    assert dev_unseen_eps.isdisjoint(train)
    assert dev_unseen_eps.isdisjoint(dev_seen)

    # (b) dev_unseen tasks never appear in train
    assert all(hl_splits.canonical_task(e) not in dev_unseen_tasks for e in train)

    # (c) sizes
    assert len(dev_unseen_tasks) == 2
    assert len(dev_seen) == 3


def test_save_load_round_trip(tmp_path):
    ids = _episode_ids()
    s = hl_splits.build_splits(ids, seed=1, n_dev_unseen_tasks=1, n_dev_seen_episodes=2)
    p = tmp_path / "splits.json"
    hl_splits.save_splits(p, s)
    assert hl_splits.load_splits(p) == s
    assert json.loads(p.read_text())["seed"] == 1
