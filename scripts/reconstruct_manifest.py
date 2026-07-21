"""Rebuild manifest.jsonl from already-extracted frames (for a Stage-B run killed before the final
manifest write). A manifest row is valid iff its frame jpg exists on disk."""
import argparse, json, pathlib
from openpi.training import robomind as rm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_file", required=True)
    ap.add_argument("--labels_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--sample-hz", type=float, default=1.0)
    a = ap.parse_args()
    records = json.loads(pathlib.Path(a.records_file).read_text())
    labels = json.loads(pathlib.Path(a.labels_file).read_text())
    out_dir = pathlib.Path(a.out_dir)
    rows = []
    kept_eps = 0
    for i, (rec, lab) in enumerate(zip(records, labels)):
        if lab.get("episode_id") not in (None, str(i)):
            continue
        try:
            samples = rm.build_samples(rec, lab["memories"], a.fps, a.sample_hz)
        except Exception:
            continue
        stem = rec["id"].replace("/", "_")
        ep_rows = []
        for s in samples:
            rel = f"frames/{stem}_{s['frame']}.jpg"
            if (out_dir / rel).exists():
                s["image"] = rel
                ep_rows.append(s)
        # only count an episode if all its sampled frames are present (a fully-assembled episode)
        if ep_rows and len(ep_rows) == len(samples):
            rows.extend(ep_rows)
            kept_eps += 1
    (out_dir / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Reconstructed {len(rows)} rows from {kept_eps} complete episodes -> {out_dir/'manifest.jsonl'}")


if __name__ == "__main__":
    main()
