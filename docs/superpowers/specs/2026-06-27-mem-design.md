# MEM: Multi-Scale Embodied Memory for openpi

Design spec for reproducing [MEM (arXiv:2603.03596)](https://arxiv.org/abs/2603.03596) within the openpi codebase.

## Overview

MEM adds dual-modality memory to VLA models:
- **Short-term video memory**: space-time separable attention in the ViT encoder, processing K historical frames with causal temporal attention
- **Long-term language memory**: LLM-compressed text summaries of past semantic events

These are orchestrated by a factorized policy:
- **pi_HL (high-level)**: predicts subtask instructions + memory updates from single frames + language memory
- **pi_LL (low-level)**: generates action chunks from short video history + subtask instruction

## Scope

- JAX/Flax implementation only
- Backbone-agnostic design (works with existing Gemma 2B + 300M, upgradeable to Gemma 3-4B)
- Configurable LLM backend for memory label generation (Claude, OpenAI, local models)
- Full HL/LL factorization as described in the paper

## Architecture

```
+------------------------------------------------------------------+
|                          Pi0MEM System                            |
|                                                                   |
|  +----------------+     +------------------------+               |
|  |  VideoViT       |     |  Language Memory        |              |
|  |  (SigLIP +      |     |  State (m_t)            |              |
|  |  temporal attn)  |     |  - text tokens          |              |
|  +--------+--------+     +-----------+------------+               |
|           |                          |                            |
|           v                          v                            |
|  +-------------------------------------------------------+       |
|  |             VLM Backbone (Gemma, configurable)         |       |
|  |           Shared between pi_HL and pi_LL               |       |
|  +------------+----------------------------+--------------+       |
|               |                            |                      |
|        +------v------+             +-------v------+               |
|        |    pi_HL     |             |    pi_LL     |              |
|        | High-Level   |             | Low-Level    |              |
|        | - subtask l  |             | - actions a  |              |
|        | - memory m'  |             | (flow match  |              |
|        +--------------+             |  or FAST)    |              |
|                                     +--------------+              |
|                                                                   |
|  +-------------------------------------------------------+       |
|  |         LLM Memory Label Pipeline (offline)            |       |
|  +-------------------------------------------------------+       |
+------------------------------------------------------------------+
```

## Component 1: VideoViT (Space-Time Separable Attention)

Modifies the existing SigLIP ViT to process multiple frames without adding new learnable parameters.

### Input

K frames per camera: `[b, K, h, w, 3]` where K=6 for pre-training, up to 18 for fine-tuning.

### Temporal Position Embedding

Sinusoidal embedding `e(t)` added to patch tokens before attention. Boundary condition: `e(0) = 0` for the current frame (t=0), ensuring K=1 exactly matches single-frame SigLIP. Uses the existing `posemb_sincos` function from `pi0.py`.

### Attention Pattern

Every 4th ViT layer (layers 4, 8, 12, ...) replaces standard spatial-only attention with divided space-time attention:

1. **Temporal attention** (first): Each patch at position p attends causally to the same patch position p across all K timesteps. Operates on `[b*n, K, d]`. Causal mask ensures each frame only sees past/current frames.
2. **Spatial attention** (second): Standard bidirectional within-frame attention on the temporally-updated tokens. Operates on `[b*K, n, d]`.

Other layers: standard spatial attention on `[b*K, n, d]` (frames processed independently).

Complexity: O(Kn^2 + nK^2) vs O(n^2 K^2) for naive joint attention.

### Token Dropping

After the last temporal attention layer, drop all past-frame tokens, keeping only the current frame's `[b, n, d]` tokens. From this point, the token count matches single-frame processing, and output feeds into the VLM backbone as normal.

### No New Parameters

Temporal attention reuses the existing QKV weight matrices. Sinusoidal temporal embeddings are computed on the fly (not learned). Gradient flows through temporal attention and updates the existing ViT weights.

### Implementation

New file: `src/openpi/models/video_vit.py`

Wraps the existing SigLIP `Module`:
- Accepts multi-frame input
- Adds temporal position embeddings before each attention layer
- Reshapes tokens for temporal attention (b*n, K, d) and spatial attention (b*K, n, d)
- Applies causal mask during temporal attention
- Drops past-frame tokens after final temporal layer
- Returns `[b, n, d]` tokens (same shape as single-frame SigLIP)

### Config

```python
@dataclasses.dataclass(frozen=True)
class VideoViTConfig:
    num_video_frames: int = 6
    video_frame_stride_seconds: float = 1.0
    temporal_attn_every_n_layers: int = 4
```

## Component 2: Language Memory

Natural-language text summarizing relevant past events. Updated by pi_HL.

### Representation

`m_t` is a string like "Placed plate in cabinet. Moved to counter. Picked up bowl." Tokenized using the existing Gemma tokenizer.

### Integration with VLM Context

Memory tokens are prepended to the prompt tokens with bidirectional attention (both memory and prompt tokens can attend to each other and to image tokens):

```
[video_tokens | memory_tokens | prompt_tokens | state_tokens | action_tokens]
```

### State Management (Inference)

- `m_0 = ""` at episode start
- pi_HL periodically updates: `m_{t+1} = pi_HL(o_t, m_t, g)`
- Updated memory replaces previous (not appended)

### Max Length

Configurable, default 128 tokens. The LLM compression pipeline explicitly minimizes length.

### Observation Extension

```python
@struct.dataclass
class Observation:
    images: dict[str, ...]           # existing
    image_masks: dict[str, ...]      # existing
    state: ...                       # existing
    tokenized_prompt: ...            # existing
    tokenized_prompt_mask: ...       # existing
    # New MEM fields:
    tokenized_memory: ... | None     # int32[b, m] - language memory m_t
    tokenized_memory_mask: ... | None  # bool[b, m]
    tokenized_subtask: ... | None    # int32[b, s] - subtask l for LL conditioning
    tokenized_subtask_mask: ... | None  # bool[b, s]
    # Video observations (for LL policy):
    video_images: dict[str, ...] | None   # float32[b, K, h, w, 3]
    video_image_masks: dict[str, ...] | None
    video_states: ... | None         # float32[b, K, s] - proprioceptive history
```

## Component 3: Factorized Policy

### pi_LL (Low-Level Policy)

Generates action chunks. Closest to existing Pi0.

**Inputs:**
- Video observations `o_{t-K:t}`: K frames per camera, processed by VideoViT -> `[b, n, d]`
- K proprioceptive state tokens: one linear projection per timestep -> `[b, K, d_expert]`
- Subtask instruction `l_{t+1}`: tokenized text from pi_HL
- Task goal `g`: existing prompt text
- Noisy actions (flow matching) or autoregressive tokens (FAST)

**Token sequence:**
```
[video_tokens | subtask_tokens | goal_tokens | state_tokens(K) | action_tokens]
```

**Output:** Flow matching velocity prediction or FAST token prediction (same as current Pi0).

**Attention mask:** Prefix-LM style. Video + subtask + goal tokens have bidirectional attention. The K state tokens and action tokens form the causal suffix — the first state token marks the AR boundary (ar_mask=True), subsequent state tokens and action tokens use ar_mask=False within their respective groups. This extends the existing pattern where a single state token marks the AR boundary.

### pi_HL (High-Level Policy)

Runs periodically (configurable, default every 3 seconds). Produces subtask + memory.

**Inputs:**
- Current observation `o_t`: single frame via standard SigLIP (not VideoViT)
- Previous memory `m_t`: tokenized text
- Task goal `g`: tokenized text

**Token sequence:**
```
[image_tokens | memory_tokens | goal_tokens | output_tokens]
```

**Output:** Autoregressive text generation using existing Gemma LM head:
```
<subtask>pick up the bowl</subtask><memory>Placed plate in cabinet.</memory>
```

**Training loss:** Standard next-token cross-entropy on output tokens.

**No new output head needed:** Uses existing LM head of Gemma.

### Inference Orchestration

```python
class MEMPolicy:
    def step(self, observation):
        if self.should_run_hl():
            subtask, memory = self.model.predict_subtask_and_memory(
                observation.current_frame, self.memory, self.goal
            )
            self.memory = memory
            self.subtask = subtask

        actions = self.model.sample_actions(
            observation.video_frames, self.subtask, self.goal
        )
        return actions
```

### Model Class

```python
class Pi0MEM(BaseModel):
    def compute_loss_ll(self, rng, obs_video, subtask, actions):
        """Low-level flow matching loss."""

    def compute_loss_hl(self, rng, obs_single, memory, goal, target_subtask, target_memory):
        """High-level next-token prediction loss."""

    def compute_loss(self, rng, observation, actions, hl_targets):
        """Combined loss: ll_weight * L_LL + hl_weight * L_HL."""

    def sample_actions(self, rng, obs_video, subtask, goal):
        """LL inference."""

    def predict_subtask_and_memory(self, obs_single, memory, goal):
        """HL inference via autoregressive generation."""
```

## Component 4: LLM Memory Compression Pipeline

Offline tool for generating language memory training labels.

### Pipeline

```
Robot episodes with subtask annotations + success/failure flags
    -> LLM compression (configurable backend)
    -> Memory labels m_t for each timestep
```

### LLM Prompt Template

```
You are generating compressed memory summaries for a robot policy.

Given the task goal and the sequence of subtask events so far, produce a
summary that retains ONLY information still relevant for future task execution.

Rules:
- Remove details about completed subtasks that don't affect future decisions
- Aggregate repeated items (e.g., "placed 3 bowls in cabinet" not individual colors)
- Remove failed attempts that were later retried successfully
- Keep spatial information relevant to navigation
- Keep counts of remaining items
- Minimize length while preserving decision-relevant information

Task goal: {goal}
Subtask events so far: {subtask_sequence_with_success_flags}

Compressed memory summary:
```

### Configurable Backend

```python
@dataclasses.dataclass
class MemoryLabelConfig:
    backend: Literal["claude", "openai", "local"] = "claude"
    model: str = "claude-sonnet-4-20250514"
    max_memory_tokens: int = 128
    batch_size: int = 32
    api_key_env: str = "ANTHROPIC_API_KEY"  # or OPENAI_API_KEY

class MemoryLabelGenerator:
    def __init__(self, config: MemoryLabelConfig): ...
    def generate_labels(self, episodes: list[Episode]) -> list[MemoryLabels]: ...
```

### CLI

```bash
uv run python scripts/generate_memory_labels.py \
    --dataset lerobot/my_dataset \
    --backend claude \
    --output data/memory_labels/
```

## Component 5: Training Pipeline

### Training Modes

**Pre-training (HL + LL joint):**
- Diverse data mixture (teleoperated demos, rollouts, vision-language tasks)
- Video context: 6 frames, 1-second stride
- Both HL and LL losses in same forward pass
- HL loss: cross-entropy on subtask + memory tokens
- LL loss: flow matching MSE (or FAST cross-entropy)

**Fine-tuning (task-specific):**
- Extends video context up to 18 frames (~54 seconds)
- Task-specific data with memory labels from LLM pipeline
- Can fine-tune LL only, HL only, or both

### Data Loader

Extends existing DataConfig:

```python
@dataclasses.dataclass
class MEMDataConfig(DataConfig):
    num_video_frames: int = 6
    video_frame_stride_seconds: float = 1.0
    memory_labels_dir: str | None = None
    subtask_labels_dir: str | None = None
    hl_training: bool = True
```

Produces video sequences by sampling K frames from each episode at the configured stride.

### Training Config

```python
@dataclasses.dataclass(frozen=True)
class Pi0MEMTrainConfig(TrainConfig):
    hl_loss_weight: float = 1.0
    ll_loss_weight: float = 1.0
```

### Loss

```python
total_loss = ll_loss_weight * L_LL + hl_loss_weight * L_HL
```

Where:
- `L_LL` = flow matching MSE: `mean((v_t - u_t)^2)` (same as Pi0)
- `L_HL` = cross-entropy on next-token prediction of subtask + memory update

## Component 6: Proprioceptive State Encoding

K proprioceptive tokens (one per video frame timestep), each projected via a linear layer:

```python
state_tokens = self.state_proj(video_states)  # [b, K, d_expert]
```

These K tokens are placed in the suffix of the LL policy sequence, replacing the current single state token.

## File Structure

```
src/openpi/models/
    video_vit.py              # VideoViT with space-time separable attention
    pi0_mem.py                # Pi0MEM model class
    pi0_mem_config.py         # Pi0MEM configuration

src/openpi/training/
    mem_data_loader.py        # Video sequence + memory label data loader
    mem_config.py             # MEM training configs (registered in config.py)

src/openpi/policies/
    mem_policy.py             # HL/LL orchestration for inference

scripts/
    generate_memory_labels.py # LLM pipeline CLI

examples/
    mem/
        README.md
```

## Testing Strategy

### Unit Tests

- `video_vit_test.py`:
  - K=1 output exactly matches single-frame SigLIP (sinusoidal e(0)=0 guarantee)
  - Correct output shapes for K>1
  - Temporal attention is causal (future frames invisible)
  - No new parameters vs base SigLIP
  - Complexity scales as O(Kn^2 + nK^2)

- `pi0_mem_test.py`:
  - HL forward pass produces valid subtask + memory text
  - LL forward pass produces correct action shapes
  - Combined loss computation is differentiable
  - Action sampling produces correct shapes

- `mem_data_loader_test.py`:
  - Video sequence extraction at correct stride
  - Memory label loading and tokenization
  - Graceful handling of episodes shorter than K frames

### Integration Tests

- End-to-end training step with fake data
- End-to-end inference with HL/LL orchestration
- Memory label generation with mock LLM backend

## Design Decisions (Paper Unspecified)

The paper does not specify these details. Our choices:

| Decision | Our choice | Rationale |
|----------|-----------|-----------|
| K proprioceptive tokens position | After video/text tokens, before actions | Matches existing state token position |
| HL decoding strategy | Configurable (greedy default) | Paper doesn't specify; greedy is simplest |
| HL inference frequency | Configurable (default 3 seconds) | Paper says "periodically" without specifying |
| Token dropping layer | After final temporal attention layer | Paper says "pass only current timestep onwards" |
| Subtask format | Tagged text `<subtask>...</subtask><memory>...</memory>` | Standard structured generation |
| k_{0,0} in attention | Global CLS token | Consistent with SigLIP architecture |
| Proprioceptive temporal attention | Included with causal mask | Natural extension of video temporal attention |
