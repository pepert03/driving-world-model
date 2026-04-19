import copy
import sys
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast

from src import networks
from src.rssm import RSSM
from src.tools import to_f32


class Dreamer(nn.Module):
    """Minimal R2-Dreamer agent: world model + Barlow Twins + actor-critic in imagination."""

    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.barlow_lambd = float(config.barlow_lambd)
        self.return_ema = networks.ReturnEMA(device=self.device)

        self.act_dim = sum(act_space.shape)
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}

        # --- World Model ---
        encoder_type = getattr(config, "encoder_type", "cnn")
        if encoder_type == "mlp":
            mlp_shapes = {k: v for k, v in shapes.items() if len(v) < 3}
            self.encoder = networks.MLPEncoder(
                mlp_shapes,
                layers=int(getattr(config, "mlp_encoder_layers", 3)),
                units=int(getattr(config, "mlp_encoder_units", 512)),
                act=config.act,
            )
        else:
            self.encoder = networks.MultiEncoder(
                shapes, config.cnn_depth, config.cnn_mults, config.cnn_kernel, config.act,
                mlp_layers=int(getattr(config, "mlp_encoder_layers", 3)),
                mlp_units=int(getattr(config, "mlp_encoder_units", 512)),
                use_mlp=bool(getattr(config, "use_mlp_obs", True)),
            )
        self.embed_size = self.encoder.out_dim
        self.rssm = RSSM(config, self.embed_size, self.act_dim)

        self.reward = networks.MLPHead(
            self.rssm.feat_size, shape=(255,), layers=config.reward_layers,
            units=config.units, dist_name="symexp_twohot", outscale=0.0,
            device=str(self.device), bin_num=255,
        )
        self.cont = networks.MLPHead(
            self.rssm.feat_size, shape=(1,), layers=config.cont_layers,
            units=config.units, dist_name="binary", outscale=1.0,
            device=str(self.device),
        )

        # R2-Dreamer projector (feat -> embed space)
        self.projector = networks.Projector(self.rssm.feat_size, self.embed_size)

        # --- Actor-Critic ---
        self.actor = networks.MLPHead(
            self.rssm.feat_size, shape=(self.act_dim,), layers=config.actor_layers,
            units=config.units, dist_name="bounded_normal", outscale=0.01,
            device=str(self.device), min_std=config.actor_min_std, max_std=config.actor_max_std,
        )
        self.value = networks.MLPHead(
            self.rssm.feat_size, shape=(255,), layers=config.critic_layers,
            units=config.units, dist_name="symexp_twohot", outscale=0.0,
            device=str(self.device), bin_num=255,
        )
        self._slow_value = copy.deepcopy(self.value)
        for p in self._slow_value.parameters():
            p.requires_grad = False
        self._slow_value_updates = 0
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)

        # Loss scales
        self._scales = {
            "dyn": float(config.dyn_scale),
            "rep": float(config.rep_scale),
            "barlow": float(config.barlow_scale),
            "rew": float(config.reward_scale),
            "con": float(config.cont_scale),
            "policy": float(config.policy_scale),
            "value": float(config.value_scale),
            "repval": float(config.repval_scale),
        }

        # Collect all trainable parameters
        modules = OrderedDict([
            ("rssm", self.rssm), ("actor", self.actor), ("value", self.value),
            ("reward", self.reward), ("cont", self.cont),
            ("encoder", self.encoder), ("projector", self.projector),
        ])
        self._named_params = OrderedDict()
        for name, module in modules.items():
            for pname, param in module.named_parameters():
                self._named_params[f"{name}.{pname}"] = param

        # Optimizer: Adam with warmup
        self._lr = float(config.lr if hasattr(config, 'lr') else 4e-5)
        self._grad_clip = float(config.grad_clip if hasattr(config, 'grad_clip') else 1000.0)
        self._warmup = int(config.warmup if hasattr(config, 'warmup') else 1000)
        self._optimizer = torch.optim.Adam(
            self._named_params.values(), lr=self._lr,
            eps=float(config.eps if hasattr(config, 'eps') else 1e-8),
        )
        self._scaler = GradScaler()
        self._sched_step = 0

        param_count = sum(p.numel() for p in self._named_params.values())
        print(f"R2-Dreamer agent: {param_count:,} parameters")

        self.train()
        self.clone_and_freeze()

    # --- Frozen copies for imagination ---

    def clone_and_freeze(self):
        def _unwrap(mod):
            """Get the underlying module if compiled (has _orig_mod)."""
            return getattr(mod, "_orig_mod", mod)

        def _freeze(src):
            src = _unwrap(src)
            dst = copy.deepcopy(src)
            for (n1, p1), (n2, p2) in zip(src.named_parameters(), dst.named_parameters()):
                p2.data = p1.data
                p2.requires_grad_(False)
            return dst

        self._frozen_encoder = _freeze(self.encoder)
        self._frozen_rssm = _freeze(self.rssm)
        self._frozen_reward = _freeze(self.reward)
        self._frozen_cont = _freeze(self.cont)
        self._frozen_actor = _freeze(self.actor)
        self._frozen_value = _freeze(self.value)
        self._frozen_slow_value = _freeze(self._slow_value)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.clone_and_freeze()
        if self.device.type == "cuda" and sys.platform != "win32":
            self._compile()
        return self

    def _compile(self):
        """Compile only the trainable modules (not frozen copies).
        Frozen copies are used for inference in act() and must stay
        uncompiled to avoid FX-tracing-into-dynamo errors."""
        mode = "default"
        self.encoder = torch.compile(self.encoder, mode=mode)
        self.rssm = torch.compile(self.rssm, mode=mode)
        self.actor = torch.compile(self.actor, mode=mode)
        self.value = torch.compile(self.value, mode=mode)
        self.reward = torch.compile(self.reward, mode=mode)
        self.cont = torch.compile(self.cont, mode=mode)
        self.projector = torch.compile(self.projector, mode=mode)


    def train(self, mode=True):
        super().train(mode)
        self._slow_value.train(False)
        return self

    def _update_slow_target(self):
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    # --- Warmup LR ---

    def _step_lr(self):
        self._sched_step += 1
        factor = min(1.0, self._sched_step / self._warmup) if self._warmup else 1.0
        for pg in self._optimizer.param_groups:
            pg["lr"] = self._lr * factor

    # --- Policy inference ---

    @torch.no_grad()
    def act(self, obs, state, eval_mode=False):
        p_obs = self.preprocess(obs)
        embed = self._frozen_encoder(p_obs)
        stoch, deter, prev_action = state["stoch"], state["deter"], state["prev_action"]
        stoch, deter, _ = self._frozen_rssm.obs_step(stoch, deter, prev_action, embed, obs["is_first"])
        feat = self._frozen_rssm.get_feat(stoch, deter)
        action_dist = self._frozen_actor(feat)
        action = action_dist.mode if eval_mode else action_dist.rsample()
        return action, TensorDict(
            {"stoch": stoch, "deter": deter, "prev_action": action},
            batch_size=state.batch_size,
        )

    @torch.no_grad()
    def get_initial_state(self, B):
        stoch, deter = self.rssm.initial(B, self.device)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        return TensorDict({"stoch": stoch, "deter": deter, "prev_action": action}, batch_size=(B,))

    @torch.no_grad()
    def preprocess(self, data):
        if "image" in data:
            data = data.clone()
            data["image"] = to_f32(data["image"]) / 255.0
        return data

    # --- Training step ---

    def update(self, replay_buffer):
        data, index, initial = replay_buffer.sample()
        p_data = self.preprocess(data)
        self._update_slow_target()

        if self.device.type == "cpu":
            (stoch, deter), metrics = self._compute_losses(p_data, initial)
        else:
            with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                (stoch, deter), metrics = self._compute_losses(p_data, initial)

        self._scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._named_params.values(), self._grad_clip)
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._step_lr()
        self._optimizer.zero_grad(set_to_none=True)

        self.clone_and_freeze()
        replay_buffer.update(index, stoch.detach(), deter.detach())
        return metrics

    def _compute_losses(self, data, initial):
        losses = {}
        metrics = {}
        B, T = data.shape

        # === World model: posterior rollout ===
        embed = self.encoder(data)
        post_stoch, post_deter, post_logit = self.rssm.observe(embed, data["action"], initial, data["is_first"])
        _, prior_logit = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)

        # === R2-Dreamer: Barlow Twins loss ===
        feat = self.rssm.get_feat(post_stoch, post_deter)
        x1 = self.projector(feat.reshape(B * T, -1))
        x2 = embed.reshape(B * T, -1).detach()
        x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
        x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)
        c = torch.mm(x1_norm.T, x2_norm) / (B * T)
        invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
        off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
        redundancy_loss = c[off_diag_mask].pow(2).sum()
        losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss

        # === Reward & Continue heads ===
        losses["rew"] = torch.mean(-self.reward(feat).log_prob(to_f32(data["reward"])))
        cont_target = 1.0 - to_f32(data["is_terminal"])
        losses["con"] = torch.mean(-self.cont(feat).log_prob(cont_target))

        # === Imagination rollout for actor-critic ===
        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
        )
        imag_feat, imag_action = self._imagine(start, self.imag_horizon + 1)
        imag_feat, imag_action = imag_feat.detach(), imag_action.detach()

        imag_reward = self._frozen_reward(imag_feat).mode()
        imag_cont = self._frozen_cont(imag_feat).mean
        imag_value = self._frozen_value(imag_feat).mode()
        imag_slow_value = self._frozen_slow_value(imag_feat).mode()

        disc = 1 - 1 / self.horizon
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(last, term, imag_reward, imag_value, imag_value, disc, self.lamb)
        ret_offset, ret_scale = self.return_ema(ret)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        # Policy loss
        policy = self.actor(imag_feat)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))

        # Value loss
        imag_value_dist = self.value(imag_feat)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        losses["value"] = torch.mean(
            weight[:, :-1].detach()
            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[:, :-1].unsqueeze(-1)
        )

        # === Replay-based value learning (gradients through world model) ===
        last_r = to_f32(data["is_last"])
        term_r = to_f32(data["is_terminal"])
        reward_r = to_f32(data["reward"])
        feat_r = self.rssm.get_feat(post_stoch, post_deter)
        boot = ret[:, 0].reshape(B, T, 1)
        value_r = self._frozen_value(feat_r).mode()
        slow_value_r = self._frozen_slow_value(feat_r).mode()
        weight_r = 1.0 - last_r
        ret_r = self._lambda_return(last_r, term_r, reward_r, value_r, boot, disc, self.lamb)
        ret_r_padded = torch.cat([ret_r, 0 * ret_r[:, -1:]], 1)
        value_dist_r = self.value(feat_r)
        losses["repval"] = torch.mean(
            weight_r[:, :-1]
            * (-value_dist_r.log_prob(ret_r_padded.detach()) - value_dist_r.log_prob(slow_value_r.detach()))[:, :-1].unsqueeze(-1)
        )

        # Total loss with scaling
        total_loss = sum(v * self._scales[k] for k, v in losses.items())
        self._scaler.scale(total_loss).backward()

        metrics.update({f"loss/{k}": v.detach() for k, v in losses.items()})
        metrics["loss/total"] = total_loss.detach()
        metrics["ret_mean"] = torch.mean(ret).detach()
        metrics["adv_mean"] = torch.mean(adv).detach()
        return (post_stoch, post_deter), metrics

    @torch.no_grad()
    def _imagine(self, start, horizon):
        feats, actions = [], []
        stoch, deter = start
        for _ in range(horizon):
            feat = self._frozen_rssm.get_feat(stoch, deter)
            action = self._frozen_actor(feat).rsample()
            feats.append(feat)
            actions.append(action)
            stoch, deter = self._frozen_rssm.img_step(stoch, deter, action)
        return torch.stack(feats, 1), torch.stack(actions, 1)

    @torch.no_grad()
    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)
