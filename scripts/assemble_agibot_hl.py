"""Assemble AgiBot HL training samples (records + memory labels + local raw video) -> manifest + frames.

Stage A first (build records from a task_info/task_<task_id>.json file):
    uv run python -c "
    import json
    from openpi.training import agibot
    records = agibot.records_from_repo('agibot-world/AgiBotWorld-Alpha', 327)
    json.dump([r.__dict__ for r in records], open('data/agibot_records.json', 'w'))
    "
Then Stage B (reuse the existing generator to produce labels):
    uv run python scripts/generate_memory_labels.py --episodes_file data/agibot_episodes.json \
        --backend openai --base_url http://localhost:8000/v1 --model Qwen/Qwen2.5-VL-72B-Instruct \
        --output data/agibot_labels.json --disable_thinking
Before Stage C, download + extract the task's observation tar(s) (head-cam video only, since
per-task tars can be up to ~48GB and only the head-cam mp4 is needed for HL frames):
    uv run python -c "
    from openpi.training import agibot
    tars = agibot.download_task_tars('agibot-world/AgiBotWorld-Alpha', 327)
    for t in tars:
        agibot.extract_task_head_videos(t, 'data/agibot_extracted')
    "
Then Stage C (this script). AgiBot ships a RAW (non-LeRobot) layout with NO meta/info.json --
after extraction, video lives at <extracted>/<episode_id>/videos/head_color.mp4 -- so this script
points at the extracted dir with an explicit --video-path-template rather than an HF repo id or a
LeRobot-layout archive. Codec is AV1, so this defaults to the PyAV decode backend (same as Galaxea):
    uv run python scripts/assemble_agibot_hl.py --records_file data/agibot_records.json \
        --labels_file data/agibot_labels.json \
        --video-root data/agibot_extracted \
        --out_dir data/agibot_hl
"""

import argparse
import json
import pathlib

from openpi.training import lerobot_hl as lh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records_file", required=True)
    p.add_argument("--labels_file", required=True)
    p.add_argument("--video-root", required=True, help="locally-extracted AgiBot head-video dir (<eid>/videos/<key>)")
    p.add_argument("--out_dir", default="data/agibot_hl")
    p.add_argument("--video-key", default="head_color.mp4")
    p.add_argument("--video-path-template", default="{episode_index}/videos/{video_key}")
    p.add_argument("--decode-backend", default="av")
    p.add_argument("--fps", type=float, default=30.0)
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
        video_path_template=args.video_path_template,
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
