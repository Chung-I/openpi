"""Generate the golden fixture for the K=1 video-encoder equivalence test.

Run this INSIDE an origin/main checkout so the reference output is produced by
origin/main's siglip.py. It builds the pi0.5-style SigLIP image encoder (same
_siglip.Module class pi0.py uses), runs one fixed image, and writes:
  pi05_encoder_ref.msgpack  - flax params
  pi05_encoder_ref_image.npy - the input image [1, 16, 16, 3]
  pi05_encoder_ref_out.npy   - encoder output
  pi05_encoder_ref_meta.json - {"siglip_sha256": ...} of the siglip.py used

Usage (from an origin/main worktree):
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/eval/gen_pi05_encoder_fixture.py <out_dir>
"""

import hashlib
import json
import pathlib
import sys

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.siglip as _siglip

SIGLIP_KWARGS = dict(
    num_classes=32, variant="mu/2", pool_type="none", scan=False, dtype_mm="float32"
)


def main(out_dir: str) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    image = np.asarray(
        jax.random.normal(jax.random.key(42), (1, 16, 16, 3)), dtype=np.float32
    )
    module = _siglip.Module(**SIGLIP_KWARGS)
    variables = module.init(jax.random.key(0), jnp.asarray(image), train=False)
    output, _ = module.apply(variables, jnp.asarray(image), train=False)

    (out / "pi05_encoder_ref.msgpack").write_bytes(
        flax.serialization.to_bytes(variables["params"])
    )
    np.save(out / "pi05_encoder_ref_image.npy", image)
    np.save(out / "pi05_encoder_ref_out.npy", np.asarray(output))

    siglip_sha = hashlib.sha256(pathlib.Path(_siglip.__file__).read_bytes()).hexdigest()
    (out / "pi05_encoder_ref_meta.json").write_text(
        json.dumps({"siglip_sha256": siglip_sha, "siglip_kwargs": SIGLIP_KWARGS}, indent=2)
    )
    print(f"wrote fixture to {out} (siglip_sha256={siglip_sha[:12]}...)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures")
