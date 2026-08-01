"""State-cond pi05: fresh modules exist, adarms_cond depends on state, prompt has no State: section."""
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0, pi0_config
from openpi.models.tokenizer import PaligemmaTokenizer


def test_config_flag_default_false():
    cfg = pi0_config.Pi0Config(pi05=True)
    assert cfg.state_cond is False


def test_state_changes_adarms_cond():
    import flax.nnx as nnx

    cfg = pi0_config.Pi0Config(pi05=True, state_cond=True, action_horizon=15)
    model = pi0.Pi0(cfg, rngs=nnx.Rngs(0))
    obs = cfg.fake_obs()
    x_t = jnp.zeros((1, cfg.action_horizon, cfg.action_dim))
    t = jnp.array([0.5])
    _, _, _, cond_a = model.embed_suffix(obs, x_t, t)
    obs2 = obs.replace(state=obs.state + 1.0)
    _, _, _, cond_b = model.embed_suffix(obs2, x_t, t)
    assert cond_a is not None
    assert not jnp.allclose(cond_a, cond_b), "adarms_cond must depend on state when state_cond=True"


def test_state_cond_false_matches_stock():
    import flax.nnx as nnx

    cfg = pi0_config.Pi0Config(pi05=True, state_cond=False, action_horizon=15)
    model = pi0.Pi0(cfg, rngs=nnx.Rngs(0))
    obs = cfg.fake_obs()
    x_t = jnp.zeros((1, cfg.action_horizon, cfg.action_dim))
    t = jnp.array([0.5])
    _, _, _, cond_a = model.embed_suffix(obs, x_t, t)
    obs2 = obs.replace(state=obs.state + 1.0)
    _, _, _, cond_b = model.embed_suffix(obs2, x_t, t)
    assert jnp.allclose(cond_a, cond_b), "stock pi05 adarms_cond is time-only"


def test_tokenizer_pi05_no_state_prompt():
    tok = PaligemmaTokenizer(max_len=48)
    with_state, _ = tok.tokenize("pick the ball", state=np.zeros(8))
    no_state, _ = tok.tokenize("pick the ball", state=None, pi05_no_state=True)
    text_with = tok._tokenizer.decode(with_state.tolist())
    text_no = tok._tokenizer.decode(no_state.tolist())
    assert "State:" in text_with
    assert "State:" not in text_no
    assert "Task: pick the ball" in text_no and "Action:" in text_no
