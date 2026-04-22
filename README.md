
# R2-Dreamer — World Models for Visual Continuous Control

Implementation of [R2-Dreamer](https://openreview.net/forum?id=Je2QqXrcQq) (Morihira et al., ICLR 2026), a World Model agent that learns a latent dynamics model of the environment and trains its policy entirely *in imagination*, without a pixel decoder.

## Results

<table>
<tr>
<th>CarRacing-v3</th>
<th>Hopper-v5</th>
<th>Walker Walk</th>
</tr>
<tr>
<td><img src="assets/car_racing.gif" style="max-width:100%;"/><br/>Best reward: <strong>941</strong></td>
<td><img src="assets/hopper_v5.gif" style="max-width:100%;"/><br/>Best reward: <strong>21711</strong></td>
<td><img src="assets/walker_walk.gif" style="max-width:100%;"/><br/>Best reward: <strong>927</strong></td>
</tr>
</table>

## Theoretical background

### World Models

Instead of learning a policy directly from environment interactions (model-free), a **World Model** learns an internal model of the environment dynamics. The agent can then *imagine* future trajectories in latent space and optimize its policy on them, dramatically improving sample efficiency.

The training loop has three phases:

1. **Experience collection** — the agent interacts with the real environment and stores transitions in a replay buffer.
2. **World model learning** — the model learns to predict future latent states, rewards and episode continuations from past experience.
3. **Imagination** — the actor-critic is trained on trajectories *dreamed* by the world model, without touching the real environment.

### Recurrent State-Space Model (RSSM)

The core of the world model is the **RSSM**, which maintains a latent state with two components:

- **Deterministic** $h_t$: a recurrent state updated by a Block-GRU.
- **Stochastic** $z_t$: 32 categorical variables with 16 classes each, sampled via Gumbel-Softmax.

The full latent state is $s_t = (h_t, z_t)$.

**Sequence model** (deterministic transition):

$$h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1})$$

**Prior** (what the model predicts without observation):

$$\hat{z}_t \sim p_\phi(\hat{z}_t \mid h_t)$$

**Posterior** (what actually happened, given the observation):

$$z_t \sim q_\phi(z_t \mid h_t, e_t), \quad e_t = \text{enc}(o_t)$$

The world model is trained by minimizing the KL divergence between posterior and prior, plus reward and continuation prediction losses:

$$\mathcal{L}_{\text{wm}} = \beta_{\text{dyn}} \underbrace{D_{\text{KL}}[\text{sg}(q_\phi) \| p_\phi]}_{\text{dynamics loss}} + \beta_{\text{rep}} \underbrace{D_{\text{KL}}[q_\phi \| \text{sg}(p_\phi)]}_{\text{representation loss}} + \mathcal{L}_{\text{reward}} + \mathcal{L}_{\text{cont}}$$

where $\text{sg}$ denotes stop-gradient. This **KL balancing** (with $\beta_{\text{dyn}} = 1.0$, $\beta_{\text{rep}} = 0.1$) encourages the prior to match the posterior rather than the other way around, preventing posterior collapse.

### R2-Dreamer: decoder-free representation

Standard DreamerV3 trains the encoder by reconstructing pixel observations through a CNN decoder. **R2-Dreamer** removes the decoder entirely and replaces it with a **Barlow Twins** redundancy-reduction loss on the latent representations:

$$\mathcal{L}_{\text{BT}} = \underbrace{\sum_i (1 - \mathcal{C}_{ii})^2}_{\text{invariance}} + \lambda_{\text{BT}} \underbrace{\sum_i \sum_{j \neq i} \mathcal{C}_{ij}^2}_{\text{redundancy reduction}}$$

where $\mathcal{C}$ is the cross-correlation matrix between projected posterior and prior embeddings. This enforces that: (1) corresponding dimensions are correlated (diagonal → 1), and (2) different dimensions are decorrelated (off-diagonal → 0).

This yields **~1.6x speedup** over DreamerV3 with comparable or better performance, since the expensive CNN decoder is eliminated.

### Actor-Critic in imagination

The actor and critic are trained entirely on imagined trajectories of length $H$ rolled out from real latent states:

**Actor objective** (maximize imagined returns + entropy):

$$\mathcal{L}_{\text{actor}} = -\mathbb{E}\left[\sum_{t=0}^{H} \left(\text{sg}(V_\lambda(s_t)) + \eta \, \mathcal{H}[\pi(a_t \mid s_t)]\right)\right]$$

**$\lambda$-returns** (bootstrapped value targets):

$$V_\lambda(s_t) = r_t + \gamma \left[(1-\lambda)\, v_\psi(s_{t+1}) + \lambda \, V_\lambda(s_{t+1})\right]$$

**Critic loss** (TwoHot symlog distribution over 255 bins):

$$\mathcal{L}_{\text{critic}} = -\mathbb{E}\left[\ln v_\psi(s_t)\Big|_{\text{sg}(V_\lambda(s_t))}\right]$$

Return targets are normalized using **exponential moving average of the 5th–95th percentile range**, a key DreamerV3 trick for stable training across diverse reward scales.

### Key techniques

| Technique | Description |
|---|---|
| **Block-GRU** | GRU with block-diagonal linear layers for memory efficiency |
| **Gumbel-Softmax** | Straight-through gradient estimator for discrete stochastic states |
| **TwoHot encoding** | Distributional value/reward prediction over 255 symlog-spaced bins |
| **Unimix** | 1% uniform mixture in categoricals to prevent distribution collapse |
| **EMA target network** | Slow-moving critic target ($\tau = 0.02$) for stable value learning |
| **ReturnEMA normalization** | Percentile-based return scaling for reward-agnostic training |

## Repository structure

```
driving-world-model/
├── main.py                        # CLI entry point (train / evaluate)
├── pyproject.toml                 # Dependencies (uv)
├── configs/
│   └── hyperparameters.yml        # All environment presets and hyperparameters
├── assets/                        # GIFs for README
├── src/
│   ├── agent.py                   # Dreamer: world model + Barlow Twins loss + actor-critic
│   ├── rssm.py                    # RSSM: Block-GRU + observe/imagine/kl_loss
│   ├── networks.py                # ConvEncoder, MLPHead, Projector, BlockLinear, ReturnEMA
│   ├── distributions.py           # OneHotDist, TwoHot, BoundedNormal, symlog
│   ├── buffer.py                  # Replay buffer (TorchRL SliceSampler)
│   ├── envs.py                    # DMC/Gymnasium wrappers + ParallelEnv
│   └── tools.py                   # Logger, config loading, utilities
├── runs/                          # Generated outputs (one folder per preset)
│   └── <preset_name>/
│       ├── config.yml             # Frozen copy of the configuration used
│       ├── best_model.pt          # Best model weights (by episodic return)
│       ├── checkpoint.pt          # Full training state (resumable)
│       ├── training.log           # Timestamped training log
│       └── graph.png              # Reward curve plot
├── scripts/
│   └── generate_video_frames.py   # Extract evaluation frames
├── DQN-Rainbow-Pixel-Control/     # Model-free baseline (separate project)
└── Report/                        # Typst presentation
```

Training is **automatically resumable**: if `checkpoint.pt` exists, the agent restores its full state and continues from the last saved step.

## Installation

### Requirements

- Python 3.11 or 3.12
- CUDA-capable GPU (recommended; ~8 GB VRAM sufficient for the 12M model)
- MuJoCo (installed automatically via `dm-control`)

### Setup

```bash
git clone https://github.com/pepert03/driving-world-model
cd driving-world-model

uv python install 3.11
uv venv --python 3.11
uv sync
```

## Usage

### Available presets

All presets are defined in `configs/hyperparameters.yml`:

| Preset | Environment | Observation | Key hyperparameters |
|---|---|---|---|
| `walker_walk` | DMC Walker Walk | 64×64 pixels | batch=32, seq=32, horizon=8, lr=4e-5 |
| `hopper_hop` | DMC Hopper Hop | 64×64 pixels | batch=16, seq=64, horizon=15, lr=4e-5 |
| `car_racing2` | CarRacing-v3 | 64×64 pixels | batch=32, seq=32, horizon=8, lr=4e-5 |
| `hopper_v5` | Hopper-v5 (Gymnasium) | state vector | batch=16, seq=32, horizon=15, lr=1e-4 |
| `cheetah_run` | DMC Cheetah Run | 64×64 pixels | batch=16, seq=64, horizon=15, lr=4e-5 |

### Training

```bash
# Train on Walker Walk (DMC, pixel observations)
uv run python main.py walker_walk --train

# Train on CarRacing (Gymnasium)
uv run python main.py car_racing2 --train

# Train on Hopper (Gymnasium, state observations)
uv run python main.py hopper_v5 --train
```

### Evaluation

Loads the best saved model and renders the agent:

```bash
uv run python main.py walker_walk
```

### TensorBoard

```bash
tensorboard --logdir runs/<preset_name>/tensorboard
```

## Hyperparameters

### World model (RSSM)

| Parameter | Value | Description |
|---|---|---|
| Deterministic state | 1024–2048 | Block-GRU hidden size |
| Stochastic state | 32 × 16 | Categorical variables × classes |
| KL free bits | 1.0 | Minimum KL before penalizing |
| $\beta_{\text{dyn}}$ / $\beta_{\text{rep}}$ | 1.0 / 0.1 | KL balancing weights |
| Barlow $\lambda$ | 0.0005 | Redundancy reduction coefficient |
| Barlow scale | 0.05 | Barlow Twins loss weight |

### Actor-Critic

| Parameter | Value | Description |
|---|---|---|
| Imagination horizon $H$ | 8–15 | Rollout length in latent space |
| Discount horizon | 333 ($\gamma \approx 0.997$) | $\gamma = 1 - 1/\text{horizon}$ |
| $\lambda$-return | 0.95 | GAE-style bootstrapping |
| Entropy coefficient $\eta$ | 3e-4 – 3e-3 | Action entropy regularization |
| Target EMA $\tau$ | 0.02 | Slow critic update rate |

### Training

| Parameter | Value | Description |
|---|---|---|
| Learning rate | 4e-5 – 3e-4 | Adam optimizer |
| Batch size | 16–32 | Sequences per update |
| Sequence length | 32–64 | Timesteps per sequence |
| Train ratio | 64–512 | Gradient steps per env step |
| Gradient clipping | 100–1000 | Max gradient norm |

## Main dependencies

- PyTorch >= 2.4, TorchRL >= 0.9
- MuJoCo >= 3.0, dm-control >= 1.0
- Gymnasium >= 1.0 (with Box2D for CarRacing)
- TensorBoard, NumPy, OpenCV

## References

- [R2-Dreamer: Redundancy-Reduced World Models](https://openreview.net/forum?id=Je2QqXrcQq) (Morihira et al., ICLR 2026)
- [DreamerV3: Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) (Hafner et al., 2023)
- [World Models](https://arxiv.org/abs/1803.10122) (Ha & Schmidhuber, 2018)
- [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230) (Zbontar et al., 2021)
