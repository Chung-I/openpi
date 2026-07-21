"""Serve a MEM DROID checkpoint with per-session K-frame history for RoboLab.

Usage:
    uv run --no-sync python scripts/serve_mem_session.py \
        --config pi0_mem_droid_k6_verify \
        --ckpt-dir /work/roboleon1295/openpi/checkpoints/pi0_mem_droid_k6_verify/pi0_mem_droid_k6_verify/9999 \
        --port 8000
"""

import dataclasses

import tyro

from openpi.policies import policy_config as _policy_config
from openpi.policies.mem_session_policy import MemSessionPolicy
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    config: str
    ckpt_dir: str
    port: int = 8000
    host: str = "127.0.0.1"


def main(args: Args) -> None:
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.ckpt_dir)
    served = MemSessionPolicy(policy, num_video_frames=train_config.model.num_video_frames)
    server = websocket_policy_server.WebsocketPolicyServer(served, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
