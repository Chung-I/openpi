import numpy as np

from openpi.models import tokenizer as _tokenizer
from openpi.models.tokenizer import FASTTokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)


def test_fast_tokenize_actions_shapes_and_postfix():
    tok = FASTTokenizer(max_len=64)
    actions = np.zeros((10, 7), dtype=np.float32)
    tokens, mask, loss_mask = tok.tokenize_actions(actions)
    assert tokens.shape == (64,)
    assert mask.shape == (64,)
    assert loss_mask.shape == (64,)
    # Loss is only on real postfix tokens, which are exactly the masked-in tokens.
    assert bool(loss_mask.any())
    assert not bool(loss_mask[~mask].any())  # no loss on padding


def test_tokenize_fast_actions_transform_writes_fields():
    from openpi import transforms

    tok = FASTTokenizer(max_len=64)
    tf = transforms.TokenizeFASTActions(tok)
    out = tf({"actions": np.zeros((10, 7), dtype=np.float32)})
    assert out["tokenized_action"].shape == (64,)
    assert out["tokenized_action_mask"].shape == (64,)
    assert out["tokenized_action_loss_mask"].shape == (64,)
