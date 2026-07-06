"""Assemble Galaxea HL training samples (records + memory labels + local video) -> manifest + frames.

Stage A first (build records from a locally-extracted Galaxea task archive):
    uv run python -c "
    import json
    from openpi.training import galaxea
    records = galaxea.records_from_local('data/galaxea_extracted', 'Turn_On_Off_The_Light_20250619_001')
    json.dump([r.__dict__ for r in records], open('data/galaxea_records.json', 'w'))
    "
Then Stage B (reuse the existing generator to produce labels):
    uv run python scripts/generate_memory_labels.py --episodes_file data/galaxea_episodes.json \
        --backend openai --base_url http://localhost:8000/v1 --model Qwen/Qwen2.5-VL-72B-Instruct \
        --output data/galaxea_labels.json --disable_thinking
Then Stage C (this script). Galaxea's head-cam video is AV1 (cv2 cannot decode it -- silent failure),
so this defaults to the PyAV decode backend; Galaxea also ships one whole task archive per repo file
(no lazy per-episode HF fetch), so this script points at the already-extracted archive dir rather than
an HF repo id:
    uv run python scripts/assemble_galaxea_hl.py --records_file data/galaxea_records.json \
        --labels_file data/galaxea_labels.json \
        --video-root data/galaxea_extracted/Turn_On_Off_The_Light_20250619_001 \
        --out_dir data/galaxea_hl
"""

import argparse
import json
import pathlib

from openpi.training import lerobot_hl as lh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records_file", required=True)
    p.add_argument("--labels_file", required=True)
    p.add_argument("--video-root", required=True, help="locally-extracted Galaxea archive dir (has meta/, videos/)")
    p.add_argument("--out_dir", default="data/galaxea_hl")
    p.add_argument("--video-key", default="observation.images.head_rgb")
    p.add_argument("--decode-backend", default="av")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--sample-hz", type=float, default=1.0)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--max-samples-per-subtask", type=int, default=None,
                   help="cap within-subtask HL samples per subtask to this many evenly-spaced frames "
                        "(counters a duration bias where long subtasks otherwise contribute samples "
                        "proportional to their length); default None = uncapped")
    p.add_argument("--sample-jitter", type=float, default=0.0,
                   help="seconds of max per-tick uniform noise added to within-subtask sampling times "
                        "(generation-time; default 0.0 = off, exact range(s, e, stride) grid)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --sample-jitter (deterministic per episode/subtask)")
    args = p.parse_args()

    records = json.loads(pathlib.Path(args.records_file).read_text())
    labels = json.loads(pathlib.Path(args.labels_file).read_text())
    lh.assemble(
        records,
        labels,
        out_dir=args.out_dir,
        video_root=args.video_root,
        video_key=args.video_key,
        sample_hz=args.sample_hz,
        max_episodes=args.max_episodes,
        fps_default=args.fps,
        decode_backend=args.decode_backend,
        max_samples_per_subtask=args.max_samples_per_subtask,
        sample_jitter=args.sample_jitter,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
