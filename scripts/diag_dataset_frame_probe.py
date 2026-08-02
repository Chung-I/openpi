# ruff: noqa
"""Dataset-frame prediction probe (Task 10 successor, serving-path vs training-drift
discriminator).

Pulls N random DROID RLDS frames exactly as the training pipeline materializes them
(same filter dict, same restructure/chunking, absolute joint-position action chunks),
sends each frame's RAW observation through one or more SERVED policies over the
websocket client (the same obs keys the RoboLab sim client sends), and compares the
predicted first-K actions against the dataset's ground-truth actions.

Interpretation (controller's ladder):
  - statecond/vlash MAE ~ baseline MAE (all healthy, near GT): the serving path is
    fine -> training drift confirmed as the closed-loop-collapse suspect.
  - statecond/vlash MAE garbage while baseline healthy: serving-path bug for the
    state-cond configs (diff the serve transform chain vs training: state
    normalization, padding to 32, tokenization).

A "null predictor" MAE (repeat current state for all K steps) is printed as the
scale reference separating "healthy" from "garbage".

Runs on a nano4 compute node (needs GCS egress + reachability of the serve nodes).
No GPU needed -- inference happens on the serve jobs' GPUs.

Usage:
  python scripts/diag_dataset_frame_probe.py \
      --frames 6 --chunk 15 --k 5 \
      --server baseline=25a-hgpn002:8000 \
      --server vlash=25a-hgpn004:8001 \
      --server statecond=25a-hgpn002:8002 \
      --out /work/roboleon1295/stage0/dataset_frame_probe.json
"""

import argparse
import json

import numpy as np

from openpi.training.droid_rlds_dataset import (
    DroidActionSpace,
    DroidRldsDataset,
    RLDSDataset,
)
from openpi_client import websocket_client_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=15, help="action chunk length to request from RLDS")
    ap.add_argument("--k", type=int, default=5, help="compare first K predicted actions")
    ap.add_argument(
        "--server",
        action="append",
        required=True,
        help="name=host:port, repeatable",
    )
    ap.add_argument("--shuffle-buffer", type=int, default=2000)
    ap.add_argument("--out", type=str, default="dataset_frame_probe.json")
    args = ap.parse_args()

    servers = {}
    for spec in args.server:
        name, hostport = spec.split("=", 1)
        host, port = hostport.rsplit(":", 1)
        servers[name] = websocket_client_policy.WebsocketClientPolicy(host=host, port=int(port))
        print(f"[probe] connected to {name} at {host}:{port}", flush=True)

    print("[probe] building RLDS dataset (streaming from GCS)...", flush=True)
    ds = DroidRldsDataset(
        data_dir="gs://gresearch/robotics",
        batch_size=1,
        datasets=(
            RLDSDataset(
                name="droid",
                version="1.0.1",
                weight=1.0,
                filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
            ),
        ),
        shuffle=True,
        action_chunk_size=args.chunk,
        shuffle_buffer_size=args.shuffle_buffer,
    )

    results = []
    it = iter(ds)
    for i in range(args.frames):
        batch = next(it)
        img = np.asarray(batch["observation"]["image"][0])
        wrist = np.asarray(batch["observation"]["wrist_image"][0])
        joint = np.asarray(batch["observation"]["joint_position"][0], dtype=np.float64)
        grip = np.asarray(batch["observation"]["gripper_position"][0], dtype=np.float64).reshape(-1)
        gt = np.asarray(batch["actions"][0], dtype=np.float64)  # (chunk, 8) absolute
        prompt = batch["prompt"][0]
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")

        obs = {
            "observation/exterior_image_1_left": img,
            "observation/wrist_image_left": wrist,
            "observation/joint_position": joint,
            "observation/gripper_position": grip,
            "prompt": prompt,
        }

        k = args.k
        # Null predictor: repeat the current state for all K steps.
        null_joint_mae = float(np.mean(np.abs(joint[None, :] - gt[:k, :7])))
        null_grip_mae = float(np.mean(np.abs(grip[0] - gt[:k, 7])))

        frame_res = {
            "frame": i,
            "prompt": prompt,
            "state_joint": joint.tolist(),
            "state_gripper": grip.tolist(),
            "gt_first_k": gt[:k].tolist(),
            "null_joint_mae": null_joint_mae,
            "null_grip_mae": null_grip_mae,
            "servers": {},
        }
        print(
            f"[probe] frame {i}: prompt={prompt!r} null_joint_mae={null_joint_mae:.5f} "
            f"null_grip_mae={null_grip_mae:.5f}",
            flush=True,
        )

        for name, client in servers.items():
            pred = np.asarray(client.infer(obs)["actions"], dtype=np.float64)  # (H, 8) absolute
            joint_mae = float(np.mean(np.abs(pred[:k, :7] - gt[:k, :7])))
            grip_mae = float(np.mean(np.abs(pred[:k, 7] - gt[:k, 7])))
            # Direction agreement of the commanded motion relative to current state.
            cos = []
            for t in range(k):
                dp = pred[t, :7] - joint
                dg = gt[t, :7] - joint
                denom = np.linalg.norm(dp) * np.linalg.norm(dg)
                cos.append(float(dp @ dg / denom) if denom > 1e-12 else float("nan"))
            frame_res["servers"][name] = {
                "pred_first_k": pred[:k].tolist(),
                "joint_mae": joint_mae,
                "grip_mae": grip_mae,
                "dir_cosine": cos,
                "finite": bool(np.isfinite(pred).all()),
            }
            print(
                f"[probe]   {name:>10}: joint_mae={joint_mae:.5f} grip_mae={grip_mae:.5f} "
                f"dir_cos_mean={np.nanmean(cos):.3f} finite={np.isfinite(pred).all()}",
                flush=True,
            )
        results.append(frame_res)

    # Summary table.
    print("\n[probe] ==== SUMMARY (mean over frames, first %d steps) ====" % args.k, flush=True)
    null_j = np.mean([r["null_joint_mae"] for r in results])
    print(f"[probe] {'null(hold-state)':>16}: joint_mae={null_j:.5f}", flush=True)
    summary = {"null_joint_mae": float(null_j), "servers": {}}
    for name in servers:
        jm = np.mean([r["servers"][name]["joint_mae"] for r in results])
        gm = np.mean([r["servers"][name]["grip_mae"] for r in results])
        dc = np.nanmean([np.nanmean(r["servers"][name]["dir_cosine"]) for r in results])
        summary["servers"][name] = {"joint_mae": float(jm), "grip_mae": float(gm), "dir_cosine": float(dc)}
        print(
            f"[probe] {name:>16}: joint_mae={jm:.5f} grip_mae={gm:.5f} dir_cos={dc:.3f}",
            flush=True,
        )

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "frames": results}, f, indent=2)
    print(f"[probe] wrote {args.out}", flush=True)
    print("PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
