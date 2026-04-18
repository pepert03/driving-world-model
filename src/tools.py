import os
import random
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import yaml
import torch
from torch import nn
from torch.nn import init as nn_init
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# ── Constants ──────────────────────────────────────────────────────

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
CONFIG = "./configs/hyperparameters.yml"
os.makedirs(RUNS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Config ─────────────────────────────────────────────────────────

def load_config(hyperparameter_set):
    """Load config from runs dir (if resuming) or from main configs file."""
    config_file = os.path.join(RUNS_DIR, hyperparameter_set, "config.yml")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    else:
        with open(CONFIG, "r") as f:
            all_config = yaml.safe_load(f)
            config = all_config[hyperparameter_set]
            os.makedirs(os.path.join(RUNS_DIR, hyperparameter_set), exist_ok=True)
            with open(config_file, "w") as f:
                yaml.dump(config, f)
            return config


def config_to_namespace(config):
    """Convert dict config to SimpleNamespace for attribute access."""
    return SimpleNamespace(**config)


# ── Plotting ───────────────────────────────────────────────────────

def save_graph(graph_file, rewards_per_episode):
    fig = plt.figure(1)
    mean_rewards = np.zeros(len(rewards_per_episode))
    for x in range(len(mean_rewards)):
        mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])
    plt.ylabel("Mean Rewards")
    plt.plot(mean_rewards)
    fig.savefig(graph_file)
    plt.close(fig)


# ── Utilities ──────────────────────────────────────────────────────

def to_np(x):
    return x.detach().cpu().numpy()


def to_f32(x):
    return x.to(dtype=torch.float32)


def to_i32(x):
    return x.to(dtype=torch.int32)


def rpad(x, n):
    """Right-pad tensor with singleton dims."""
    return x.reshape(*x.shape, *([1] * n))


def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weight_init_(m, fan_type="fan_in"):
    if isinstance(m, nn.RMSNorm):
        with torch.no_grad():
            m.weight.fill_(1.0)
    elif isinstance(m, nn.Linear):
        if fan_type == "in":
            fan_type = "fan_in"
        elif fan_type == "out":
            fan_type = "fan_out"
        fan = nn_init._calculate_correct_fan(m.weight, fan_type)
        scale = fan ** -0.5
        with torch.no_grad():
            m.weight.uniform_(-scale, scale)
            if m.bias is not None:
                m.bias.zero_()


def tensorstats(tensor, prefix):
    return {
        f"{prefix}_mean": torch.mean(tensor),
        f"{prefix}_std": torch.std(tensor),
    }


class Every:
    """Returns number of triggers since last call."""
    def __init__(self, every):
        self._every = int(every)
        self._last = 0

    def __call__(self, step):
        if not self._every:
            return 0
        count = int(step // self._every - self._last // self._every)
        self._last = step
        return count


class Once:
    def __init__(self):
        self._done = False

    def __call__(self):
        if self._done:
            return False
        self._done = True
        return True
