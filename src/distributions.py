import torch
from torch import distributions as torchd
from torch.nn import functional as F


def to_f32(x):
    return x.to(dtype=torch.float32)


def to_i32(x):
    return x.to(dtype=torch.int32)


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


# --- Categorical with Gumbel-Softmax (for RSSM stochastic state) ---

class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits, unimix_ratio=0.0):
        probs = F.softmax(to_f32(logits), dim=-1)
        uniform = unimix_ratio / probs.shape[-1]
        probs = probs * (1.0 - unimix_ratio) + torch.ones_like(probs) * uniform
        logits = torch.log(probs)
        super().__init__(logits=logits)

    @property
    def mode(self):
        _mode = F.one_hot(torch.argmax(self.logits, axis=-1), self.logits.shape[-1])
        return _mode.detach() + self.logits - self.logits.detach()

    def rsample(self, sample_shape=(), temperature=1.0):
        return F.gumbel_softmax(self.logits, tau=temperature, hard=True, dim=-1)

    def sample(self, **kwargs):
        raise NotImplementedError


# --- TwoHot distribution for scalar prediction (reward, value) ---

class TwoHot:
    def __init__(self, logits, bins, squash=None, unsquash=None):
        self.logits = to_f32(logits)
        self.bins = bins
        self.probs = F.softmax(self.logits, dim=-1)
        self.squash = squash if squash is not None else (lambda x: x)
        self.unsquash = unsquash if unsquash is not None else (lambda x: x)

    def mode(self):
        n = self.logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1, p2, p3 = self.probs[..., :m], self.probs[..., m:m+1], self.probs[..., m+1:]
            b1, b2, b3 = self.bins[..., :m], self.bins[..., m:m+1], self.bins[..., m+1:]
            wavg = (p2 * b2).sum(-1, keepdim=True) + ((p1 * b1).flip(-1) + (p3 * b3)).sum(-1, keepdim=True)
            return self.unsquash(wavg)
        p1, p2 = self.probs[..., :n//2], self.probs[..., n//2:]
        b1, b2 = self.bins[..., :n//2], self.bins[..., n//2:]
        wavg = ((p1 * b1).flip(-1) + (p2 * b2)).sum(-1, keepdim=True)
        return self.unsquash(wavg)

    def log_prob(self, target):
        assert target.dtype == self.probs.dtype
        target = target.squeeze(-1)
        target_squashed = self.squash(target).detach()
        below = to_i32(self.bins <= target_squashed.unsqueeze(-1)).sum(-1) - 1
        above = len(self.bins) - to_i32(self.bins > target_squashed.unsqueeze(-1)).sum(-1)
        below = torch.clamp(below, 0, len(self.bins) - 1)
        above = torch.clamp(above, 0, len(self.bins) - 1)
        equal = below == above
        dist_to_below = torch.where(equal, torch.ones_like(target), (self.bins[below] - target_squashed).abs())
        dist_to_above = torch.where(equal, torch.ones_like(target), (self.bins[above] - target_squashed).abs())
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        oh_below = to_f32(F.one_hot(below, num_classes=len(self.bins)))
        oh_above = to_f32(F.one_hot(above, num_classes=len(self.bins)))
        mixed_target = oh_below * weight_below.unsqueeze(-1) + oh_above * weight_above.unsqueeze(-1)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        return (mixed_target * log_pred).sum(-1)


# --- Simple distributions ---

class MSEDist:
    def __init__(self, mode):
        self._mode = to_f32(mode)

    def mode(self):
        return self._mode

    @property
    def mean(self):
        return self._mode

    def log_prob(self, value):
        distance = (self._mode - value) ** 2
        return -distance.sum(list(range(len(distance.shape)))[2:])


class SymlogDist:
    def __init__(self, mode):
        self._mode = to_f32(mode)

    def mode(self):
        return symexp(self._mode)

    @property
    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        distance = (self._mode - symlog(value)) ** 2.0
        distance = torch.where(distance < 1e-8, 0, distance)
        return -distance.sum(list(range(len(distance.shape)))[2:])


# --- Distribution constructors (used by MLPHead) ---

def bounded_normal(x, min_std, max_std, **kwargs):
    mean, std = torch.chunk(x, 2, dim=-1)
    std = (max_std - min_std) * torch.sigmoid(std + 2.0) + min_std
    dist = torchd.normal.Normal(torch.tanh(to_f32(mean)), to_f32(std))
    return torchd.independent.Independent(dist, 1)


def binary(logits, **kwargs):
    return torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=to_f32(logits)), 1)


def symexp_twohot(logits, bin_num, **kwargs):
    if bin_num % 2 == 1:
        half = torch.linspace(-20, 0, (bin_num - 1) // 2 + 1, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.cat([half, -half[:-1].flip(0)], 0)
    else:
        half = torch.linspace(-20, 0, bin_num // 2, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.cat([half, -half.flip(0)], 0)
    return TwoHot(to_f32(logits), bins)


def symlog_mse(logits, **kwargs):
    return SymlogDist(to_f32(logits))


def mse(logits, **kwargs):
    return MSEDist(to_f32(logits))


# --- KL divergence for categorical logits ---

def kl(logits_left, logits_right):
    logprob_left = torch.log_softmax(logits_left, -1)
    logprob_right = torch.log_softmax(logits_right, -1)
    prob = torch.softmax(logits_left, -1)
    return (prob * (logprob_left - logprob_right)).sum(-1)
