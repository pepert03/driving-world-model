import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

import src.distributions as dists
from src.tools import weight_init_


class LambdaLayer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class BlockLinear(nn.Module):
    """Block-wise linear layer for efficient grouped computation."""

    def __init__(self, in_ch, out_ch, blocks, outscale=1.0):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.blocks = int(blocks)
        # (O/G, I/G, G)
        self.weight = nn.Parameter(torch.empty(out_ch // blocks, in_ch // blocks, blocks))
        self.bias = nn.Parameter(torch.empty(out_ch))

    def forward(self, x):
        batch_shape = x.shape[:-1]
        x = x.view(*batch_shape, self.blocks, self.in_ch // self.blocks)
        x = torch.einsum("...gi,oig->...go", x, self.weight)
        x = x.reshape(*batch_shape, self.out_ch)
        return x + self.bias


class Conv2dSamePad(nn.Conv2d):
    """Conv2d with TensorFlow-style SAME padding."""

    def _calc_same_pad(self, i, k, s, d):
        i_div_s_ceil = (i + s - 1) // s
        return max((i_div_s_ceil - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self._calc_same_pad(ih, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = self._calc_same_pad(iw, self.kernel_size[1], self.stride[1], self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class RMSNorm2D(nn.RMSNorm):
    """RMSNorm for (B, C, H, W) tensors, applied over channel dim."""

    def __init__(self, ch, eps=1e-3, dtype=None):
        super().__init__(ch, eps=eps, dtype=dtype)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvEncoder(nn.Module):
    def __init__(self, input_shape, depth, mults, kernel_size, act="SiLU", norm=True):
        super().__init__()
        act_fn = getattr(nn, act)
        h, w, input_ch = input_shape
        depths = tuple(int(depth) * int(m) for m in mults)
        in_dim = input_ch
        layers = []
        for d in depths:
            layers.append(Conv2dSamePad(in_dim, d, kernel_size, stride=1, bias=True))
            layers.append(nn.MaxPool2d(2, 2))
            if norm:
                layers.append(RMSNorm2D(d, eps=1e-04, dtype=torch.float32))
            layers.append(act_fn())
            in_dim = d
            h, w = h // 2, w // 2
        self.out_dim = depths[-1] * h * w
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        # (B, T, H, W, C)
        obs = obs - 0.5
        x = obs.reshape(-1, *obs.shape[-3:])
        x = x.permute(0, 3, 1, 2)  # (B*T, C, H, W)
        x = self.layers(x)
        x = x.reshape(x.shape[0], -1)
        return x.reshape(*obs.shape[:-3], x.shape[-1])


class MultiEncoder(nn.Module):
    """Encoder that handles image observations via CNN."""

    def __init__(self, shapes, depth, mults, kernel_size, act="SiLU", norm=True):
        super().__init__()
        cnn_shapes = {k: v for k, v in shapes.items() if len(v) == 3}
        assert cnn_shapes, "MultiEncoder requires at least one image observation"
        input_ch = sum(v[-1] for v in cnn_shapes.values())
        input_shape = tuple(cnn_shapes.values())[0][:2] + (input_ch,)
        self.cnn_keys = list(cnn_shapes.keys())
        self.encoder = ConvEncoder(input_shape, depth, mults, kernel_size, act, norm)
        self.out_dim = self.encoder.out_dim
        self.apply(weight_init_)

    def forward(self, obs):
        x = torch.cat([obs[k] for k in self.cnn_keys], dim=-1)
        return self.encoder(x)


class MLP(nn.Module):
    def __init__(self, inp_dim, layers, units, act="SiLU", norm=True, symlog_inputs=False):
        super().__init__()
        self._symlog_inputs = symlog_inputs
        act_fn = getattr(nn, act)
        net = []
        for i in range(layers):
            net.append(nn.Linear(inp_dim, units, bias=True))
            net.append(nn.RMSNorm(units, eps=1e-04, dtype=torch.float32))
            net.append(act_fn())
            inp_dim = units
        self.layers = nn.Sequential(*net)
        self.out_dim = units

    def forward(self, x):
        if self._symlog_inputs:
            x = dists.symlog(x)
        return self.layers(x)


class MLPHead(nn.Module):
    """MLP followed by a distribution head."""

    def __init__(self, inp_dim, shape, layers, units, dist_name, act="SiLU", norm=True,
                 outscale=0.0, device="cuda:0", **dist_kwargs):
        super().__init__()
        self.mlp = MLP(inp_dim, layers, units, act, norm)
        self._dist_name = dist_name
        self._dist_fn = getattr(dists, dist_name)

        if dist_name == "bounded_normal":
            self.last = nn.Linear(self.mlp.out_dim, shape[0] * 2, bias=True)
            self._kwargs = {k: v for k, v in dist_kwargs.items() if k in ("min_std", "max_std")}
        elif dist_name == "symexp_twohot":
            self.last = nn.Linear(self.mlp.out_dim, shape[0], bias=True)
            self._kwargs = {"bin_num": dist_kwargs.get("bin_num", 255)}
        elif dist_name == "binary":
            self.last = nn.Linear(self.mlp.out_dim, shape[0], bias=True)
            self._kwargs = {}
        else:
            self.last = nn.Linear(self.mlp.out_dim, shape[0], bias=True)
            self._kwargs = {}

        self.mlp.apply(weight_init_)
        self.last.apply(weight_init_)
        if outscale != 1.0:
            with torch.no_grad():
                self.last.weight.mul_(outscale)

    def forward(self, x):
        return self._dist_fn(self.last(self.mlp(x)), **self._kwargs)


class Projector(nn.Module):
    """Linear projection for R2-Dreamer Barlow Twins loss."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim, bias=False)
        self.apply(weight_init_)

    def forward(self, x):
        return self.w(x)


class ReturnEMA(nn.Module):
    """Running percentile normalization for returns."""

    def __init__(self, device, alpha=1e-2):
        super().__init__()
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)
        self.register_buffer("ema_vals", torch.zeros(2, dtype=torch.float32, device=device))

    def __call__(self, x):
        x_quantile = torch.quantile(torch.flatten(x.detach()), self.range)
        self.ema_vals.copy_(self.alpha * x_quantile.detach() + (1 - self.alpha) * self.ema_vals)
        scale = torch.clip(self.ema_vals[1] - self.ema_vals[0], min=1.0)
        offset = self.ema_vals[0]
        return offset.detach(), scale.detach()
