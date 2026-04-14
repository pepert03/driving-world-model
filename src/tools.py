import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import init as nn_init
from torch.utils.tensorboard import SummaryWriter


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


def weight_init_(m, fan_type="in"):
    if isinstance(m, nn.RMSNorm):
        with torch.no_grad():
            m.weight.fill_(1.0)
    elif isinstance(m, nn.Linear):
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
        f"{prefix}_min": torch.min(tensor),
        f"{prefix}_max": torch.max(tensor),
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


class Logger:
    def __init__(self, logdir):
        self._logdir = logdir
        self._writer = SummaryWriter(str(logdir), max_queue=1000)
        self._scalars = {}
        self._step = 0
        self._last_time = time.time()
        self._last_step = 0

    def scalar(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        self._scalars[name] = float(value)

    def video(self, name, frames):
        # frames: (B, T, H, W, C) uint8
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        # tensorboard expects (N, T, C, H, W)
        if frames.dim() == 5 and frames.shape[-1] in (1, 3):
            frames = frames.permute(0, 1, 4, 2, 3)
        self._writer.add_video(name, frames, self._step, fps=15)

    def write(self, step, fps=False):
        self._step = step
        for name, value in self._scalars.items():
            self._writer.add_scalar(name, value, step)
        if fps:
            now = time.time()
            dt = now - self._last_time
            if dt > 0:
                self._writer.add_scalar("perf/fps", (step - self._last_step) / dt, step)
            self._last_time = now
            self._last_step = step
        self._scalars.clear()
        self._writer.flush()

    def log_hydra_config(self, config):
        from omegaconf import OmegaConf
        config_str = OmegaConf.to_yaml(config)
        self._writer.add_text("config", f"```\n{config_str}\n```", 0)
        with open(self._logdir / "config.yaml", "w") as f:
            f.write(config_str)
