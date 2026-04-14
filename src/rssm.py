import torch
from torch import distributions as torchd
from torch import nn

import src.distributions as dists
from src.networks import BlockLinear, LambdaLayer
from src.tools import rpad, weight_init_


class Deter(nn.Module):
    """Block-GRU deterministic transition."""

    def __init__(self, deter, stoch, act_dim, hidden, blocks, dyn_layers, act="SiLU"):
        super().__init__()
        self.blocks = int(blocks)
        act_fn = getattr(nn, act)

        self._dyn_in0 = nn.Sequential(
            nn.Linear(deter, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act_fn()
        )
        self._dyn_in1 = nn.Sequential(
            nn.Linear(stoch, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act_fn()
        )
        self._dyn_in2 = nn.Sequential(
            nn.Linear(act_dim, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act_fn()
        )

        self._dyn_hid = nn.Sequential()
        in_ch = (3 * hidden + deter // self.blocks) * self.blocks
        for i in range(int(dyn_layers)):
            self._dyn_hid.add_module(f"dyn_hid_{i}", BlockLinear(in_ch, deter, self.blocks))
            self._dyn_hid.add_module(f"norm_{i}", nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32))
            self._dyn_hid.add_module(f"act_{i}", act_fn())
            in_ch = deter

        self._dyn_gru = BlockLinear(in_ch, 3 * deter, self.blocks)
        self.flat2group = lambda x: x.reshape(*x.shape[:-1], self.blocks, -1)
        self.group2flat = lambda x: x.reshape(*x.shape[:-2], -1)

    def forward(self, stoch, deter, action):
        B = action.shape[0]
        stoch = stoch.reshape(B, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()

        x0 = self._dyn_in0(deter)
        x1 = self._dyn_in1(stoch)
        x2 = self._dyn_in2(action)

        x = torch.cat([x0, x1, x2], -1)
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)
        x = self.group2flat(torch.cat([self.flat2group(deter), x], -1))
        x = self._dyn_hid(x)
        x = self._dyn_gru(x)

        gates = torch.chunk(self.flat2group(x), 3, dim=-1)
        reset, cand, update = (self.group2flat(g) for g in gates)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * deter


class RSSM(nn.Module):
    def __init__(self, config, embed_size, act_dim):
        super().__init__()
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        self._discrete = int(config.discrete)
        self._unimix_ratio = float(config.unimix_ratio)
        self._blocks = int(config.blocks)
        act = getattr(nn, config.act)

        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter

        self._deter_net = Deter(
            self._deter, self.flat_stoch, act_dim, self._hidden,
            blocks=self._blocks, dyn_layers=config.dyn_layers, act=config.act,
        )

        # Posterior network: deter + embed -> stoch logits
        self._obs_net = nn.Sequential()
        inp_dim = self._deter + embed_size
        for i in range(int(config.obs_layers)):
            self._obs_net.add_module(f"obs_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._obs_net.add_module(f"obs_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._obs_net.add_module(f"obs_a_{i}", act())
            inp_dim = self._hidden
        self._obs_net.add_module("obs_logit", nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
        self._obs_net.add_module("obs_reshape", LambdaLayer(
            lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)
        ))

        # Prior network: deter -> stoch logits
        self._img_net = nn.Sequential()
        inp_dim = self._deter
        for i in range(int(config.img_layers)):
            self._img_net.add_module(f"img_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._img_net.add_module(f"img_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._img_net.add_module(f"img_a_{i}", act())
            inp_dim = self._hidden
        self._img_net.add_module("img_logit", nn.Linear(inp_dim, self._stoch * self._discrete))
        self._img_net.add_module("img_reshape", LambdaLayer(
            lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)
        ))

        self.apply(weight_init_)

    def initial(self, batch_size, device):
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=device)
        return stoch, deter

    def observe(self, embed, action, initial, reset):
        """Posterior rollout using observations. Returns stochs, deters, logits."""
        L = action.shape[1]
        stoch, deter = initial
        stochs, deters, logits = [], [], []
        for i in range(L):
            stoch, deter, logit = self.obs_step(stoch, deter, action[:, i], embed[:, i], reset[:, i])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
        return torch.stack(stochs, 1), torch.stack(deters, 1), torch.stack(logits, 1)

    def obs_step(self, stoch, deter, prev_action, embed, reset):
        """Single posterior step."""
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        prev_action = torch.where(rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action)

        deter = self._deter_net(stoch, deter, prev_action)
        logit = self._obs_net(torch.cat([deter, embed], dim=-1))
        stoch = self.get_dist(logit).rsample()
        return stoch, deter, logit

    def img_step(self, stoch, deter, prev_action):
        """Single prior step (no observation)."""
        deter = self._deter_net(stoch, deter, prev_action)
        stoch, _ = self.prior(deter)
        return stoch, deter

    def prior(self, deter):
        logit = self._img_net(deter)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def get_feat(self, stoch, deter):
        """Concatenate flattened stoch with deter."""
        stoch_flat = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        return torch.cat([stoch_flat, deter], -1)

    def get_dist(self, logit):
        return torchd.independent.Independent(dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        rep_loss = dists.kl(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = dists.kl(post_logit.detach(), prior_logit).sum(-1)
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        return dyn_loss, rep_loss
