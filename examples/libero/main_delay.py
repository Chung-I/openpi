"""LIBERO evaluation with emulated inference delay: sync / naive / RTC arms.

Extends examples/libero/main.py with the delay-emulation protocol used in the
kinetix reference eval (and our RoboLab runner): a chunk requested at step T
with the observation from T only starts executing at T+delay -- the first
`delay` steps execute the tail of the previous chunk. The RTC arm additionally
sends rtc/* keys so the server inpaints the new chunk against the committed
prefix (arXiv 2506.07339; see policies/policy.py).

Protocol per cycle (execute_horizon = E, delay = d, chunk horizon = H):
  - at a chunk switch, request with the CURRENT observation
  - execute prev_chunk[E:E+d] (the already-committed overlap), then new[d:E]
  - for RTC the server aligns the previous chunk by `executed` = E steps.
A fresh rtc env_id per episode keeps the server's prefix cache episode-local.

Results append to a JSONL file (one line per episode), so runs are resumable
by re-running with --skip-done.
"""

import collections
import dataclasses
import json
import logging
import pathlib

import numpy as np
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224

    arm: str = "sync"  # sync | naive | rtc
    delay: int = 0  # inference delay in control steps
    execute_horizon: int = 5  # steps between requests (stock replan_steps=5)

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    seed: int = 7

    out_path: str = "data/libero/delay_results.jsonl"
    skip_done: bool = True  # skip (task, episode) pairs already in out_path


class DelayedChunkExecutor:
    """Client-side emulation of async inference timing (single env, stepped sim)."""

    def __init__(self, client, arm: str, delay: int, execute_horizon: int, env_id: int):
        assert arm in ("sync", "naive", "rtc"), arm
        if arm == "sync":
            assert delay == 0, "sync arm is delay-0 by definition"
        self.client = client
        self.arm = arm
        self.delay = delay
        self.k = execute_horizon
        assert self.delay <= self.k, "delay must fit in the execute window"
        self.env_id = env_id
        self.chunk = None  # remaining actions committed for execution
        self.prev_full = None  # previous full chunk (request frame)

    def act(self, element: dict) -> np.ndarray:
        """Returns the next action; requests a new chunk when the buffer empties."""
        if self.chunk is None or len(self.chunk) == 0:
            if self.arm == "rtc":
                element = dict(element)
                element["rtc/mode"] = "rtc"
                element["rtc/env_id"] = self.env_id
                element["rtc/inference_delay"] = self.delay
                element["rtc/executed"] = self.k
                # reference eval convention: prefix_attention_horizon = H - E,
                # provided server-side as the default; sent explicitly once we
                # know H (after the first response).
                if self.prev_full is not None:
                    element["rtc/prefix_attention_horizon"] = len(self.prev_full) - self.k
            full = np.asarray(self.client.infer(element)["actions"])
            assert len(full) >= self.k + self.delay, (
                f"chunk horizon {len(full)} too short for E={self.k} d={self.delay}"
            )
            if self.prev_full is None or self.delay == 0:
                # first chunk of the episode (or no delay): execute from index 0
                committed = full[: self.k]
            else:
                # overlap: the previous chunk's actions [E, E+d) are already
                # committed (in flight during inference); then the new chunk
                # takes over at its own index d (request frame = obs frame).
                committed = np.concatenate([self.prev_full[self.k : self.k + self.delay], full[self.delay : self.k]])
            self.prev_full = full
            self.chunk = collections.deque(committed)
        return self.chunk.popleft()


def eval_libero(args: Args) -> None:
    # deferred imports: keep DelayedChunkExecutor importable (and testable)
    # without the libero/robosuite stack installed
    from libero.libero import benchmark
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as _websocket_client_policy

    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    max_steps = MAX_STEPS[args.task_suite_name]

    out = pathlib.Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.skip_done and out.exists():
        for line in out.open():
            try:
                r = json.loads(line)
                done.add((r["task_id"], r["episode"]))
            except json.JSONDecodeError:
                pass

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    episode_counter = 0
    total, wins = 0, 0
    for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for episode_idx in range(args.num_trials_per_task):
            episode_counter += 1
            if (task_id, episode_idx) in done:
                continue
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            executor = DelayedChunkExecutor(
                client, args.arm, args.delay, args.execute_horizon, env_id=episode_counter
            )
            t = 0
            success = False
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, _, done_flag, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    element = {
                        "observation/image": image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                        ),
                        "observation/wrist_image": image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                        ),
                        "observation/state": np.concatenate(
                            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                        "prompt": str(task_description),
                    }
                    action = executor.act(element)
                    obs, _, done_flag, _ = env.step(action.tolist())
                    if done_flag:
                        success = True
                        break
                    t += 1
                except Exception:
                    logging.exception("episode aborted")
                    break
            total += 1
            wins += int(success)
            with out.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "suite": args.task_suite_name,
                            "arm": args.arm,
                            "delay": args.delay,
                            "execute_horizon": args.execute_horizon,
                            "task_id": task_id,
                            "episode": episode_idx,
                            "success": success,
                        }
                    )
                    + "\n"
                )
            logging.info(f"[{args.arm} d{args.delay}] {wins}/{total} ({100 * wins / max(total, 1):.1f}%)")
        env.close()
    print(f"FINAL {args.arm} d{args.delay} {args.task_suite_name}: {wins}/{total}")


def _get_libero_env(task, resolution, seed):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """Copied from robosuite: convert quaternion (x,y,z,w) to axis-angle."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
