import argparse
import itertools
import logging
import os
import threading
import time
import warnings
from datetime import datetime
from types import SimpleNamespace

# Suppress noisy third-party warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message=".*Mismatch dtype between input and weight.*")
warnings.filterwarnings("ignore", message=".*Box.*precision lowered.*")
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.ERROR)
os.environ["TORCHINDUCTOR_LOG_LEVEL"] = "ERROR"

import numpy as np
import torch
from tensordict import TensorDict
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from src.agent import Dreamer
from src.buffer import Buffer
from src.envs import make_parallel_envs, make_eval_env
from src.tools import (
    load_config,
    config_to_namespace,
    device,
    set_seed_everywhere,
    save_graph,
    RUNS_DIR,
    DATE_FORMAT,
    Every,
    Once,
)


def _bg_update_fn(agent, replay_buffer, num_updates, result):
    """Background thread: runs multiple gradient steps."""
    try:
        all_metrics = {}
        last_stoch = last_deter = last_index = None
        for _ in range(num_updates):
            data, index, initial = replay_buffer.sample()
            p_data = agent.preprocess(data)
            agent._update_slow_target()
            if agent.device.type == "cpu":
                (stoch, deter), metrics = agent._compute_losses(p_data, initial)
            else:
                with autocast(device_type=agent.device.type, dtype=torch.bfloat16):
                    (stoch, deter), metrics = agent._compute_losses(p_data, initial)
            agent._scaler.unscale_(agent._optimizer)
            torch.nn.utils.clip_grad_norm_(
                agent._named_params.values(), agent._grad_clip
            )
            agent._scaler.step(agent._optimizer)
            agent._scaler.update()
            agent._step_lr()
            agent._optimizer.zero_grad(set_to_none=True)
            # Update buffer with posterior states from last sample
            if last_stoch is not None and last_index is not None:
                replay_buffer.update(last_index, last_stoch, last_deter)
            last_stoch = stoch.detach()
            last_deter = deter.detach()
            last_index = index
            all_metrics = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in metrics.items()
            }
        result["stoch"] = last_stoch.cpu() if last_stoch is not None else None
        result["deter"] = last_deter.cpu() if last_deter is not None else None
        result["index"] = last_index
        result["metrics"] = all_metrics
        result["num_updates"] = num_updates
        result["success"] = True
    except Exception as e:
        import traceback

        result["error"] = traceback.format_exc()


def _obs_to_td(obs, device):
    """Convert a single-step env obs dict to a TensorDict with shape (1, ...)."""
    tensors = {}
    for k, v in obs.items():
        if isinstance(v, torch.Tensor):
            t = v.to(device).unsqueeze(0)
        else:
            t = torch.as_tensor(
                np.array(v) if isinstance(v, np.ndarray) else v, device=device
            ).unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        tensors[k] = t
    return TensorDict(tensors, batch_size=(1,))


def _candidate_state_keys(key):
    """Generate likely key variants across old/new module naming schemes."""
    bases = [key]
    if ".encoder.layers." in key:
        bases.append(key.replace(".encoder.layers.", "._cnn.layers."))

    out = []
    seen = set()
    for base in bases:
        candidates = [base]
        parts = base.split(".", 1)
        if len(parts) == 2:
            candidates.append(f"{parts[0]}._orig_mod.{parts[1]}")
        if "._orig_mod." in base:
            candidates.append(base.replace("._orig_mod.", "."))

        for cand in candidates:
            if cand not in seen:
                out.append(cand)
                seen.add(cand)
    return out


def _remap_state_dict(saved_sd, model_sd):
    """Remap legacy keys and keep only parameters that match by name and shape."""
    remapped = {}
    skipped = []
    for key, value in saved_sd.items():
        matched = None
        for cand in _candidate_state_keys(key):
            if cand in model_sd and tuple(model_sd[cand].shape) == tuple(value.shape):
                matched = cand
                break
        if matched is not None:
            remapped[matched] = value
        else:
            skipped.append(key)
    return remapped, skipped


def _has_mlp_encoder_keys(state_dict):
    return any(
        "encoder._mlp." in k or "encoder._orig_mod._mlp." in k
        for k in state_dict.keys()
    )


def _has_cnn_encoder_keys(state_dict):
    return any(
        "encoder._cnn.layers." in k or "encoder.encoder.layers." in k
        for k in state_dict.keys()
    )


def _with_compat_cfg(cfg, state_dict):
    """Tune runtime config for legacy checkpoints while preserving explicit user settings."""
    cfg_compat = SimpleNamespace(**vars(cfg))
    if not hasattr(cfg_compat, "use_mlp_obs"):
        cfg_compat.use_mlp_obs = True
    if (
        cfg_compat.use_mlp_obs
        and _has_cnn_encoder_keys(state_dict)
        and not _has_mlp_encoder_keys(state_dict)
    ):
        cfg_compat.use_mlp_obs = False
        print(
            "Detected legacy CNN-only checkpoint. Disabling MLP observation encoder for compatibility."
        )
    return cfg_compat


def _load_state_dict_compat(module, saved_sd, label, min_loaded_ratio=70.0):
    """Best-effort state_dict loading with key remapping for backwards compatibility."""
    model_sd = module.state_dict()
    remapped, skipped = _remap_state_dict(saved_sd, model_sd)
    incompatible = module.load_state_dict(remapped, strict=False)

    loaded_numel = sum(v.numel() for v in remapped.values())
    target_numel = sum(v.numel() for v in model_sd.values())
    loaded_ratio = (loaded_numel / max(1, target_numel)) * 100.0

    if skipped:
        print(
            f"[{label}] Skipped {len(skipped)} incompatible checkpoint tensors (name/shape mismatch)."
        )
    if incompatible.missing_keys:
        print(
            f"[{label}] Missing {len(incompatible.missing_keys)} model tensors after remap."
        )
    if incompatible.unexpected_keys:
        print(
            f"[{label}] Unexpected {len(incompatible.unexpected_keys)} checkpoint tensors after remap."
        )
    print(
        f"[{label}] Loaded {len(remapped)}/{len(model_sd)} tensors ({loaded_ratio:.1f}% params)."
    )

    if loaded_ratio < float(min_loaded_ratio):
        raise RuntimeError(
            f"Checkpoint appears incompatible with current model ({loaded_ratio:.1f}% params matched, min {float(min_loaded_ratio):.1f}%). "
            "Please evaluate with the same hyperparameter set used for training or retrain."
        )


class R2DreamerAgent:

    def __init__(self, hyperparameter_set):
        self.hyperparameter_set = hyperparameter_set
        self.config = load_config(hyperparameter_set)
        self.cfg = config_to_namespace(self.config)

        # Paths
        run_dir = os.path.join(RUNS_DIR, hyperparameter_set)
        os.makedirs(run_dir, exist_ok=True)
        self.LOG_FILE = os.path.join(run_dir, "training.log")
        self.MODEL_FILE = os.path.join(run_dir, "best_model.pt")
        self.CHECKPOINT_FILE = os.path.join(run_dir, "checkpoint.pt")
        self.GRAPH_FILE = os.path.join(run_dir, "graph.png")
        self.TB_DIR = os.path.join(run_dir, "tensorboard")
        self.EVAL_DIR = os.path.join(run_dir, "eval")

    def run(self, is_training=True, render=False):
        set_seed_everywhere(0)
        cfg = self.cfg

        # Add device to config for the agent
        cfg.device = str(device)

        if is_training:
            self._train(cfg)
        else:
            self._eval(cfg, render)

    # ── Training ───────────────────────────────────────────────────

    def _train(self, cfg):
        # Enable TF32 for ~30% faster matmul on Ampere/Ada GPUs (negligible accuracy loss)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        start_time = datetime.now()
        log_msg = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_msg)
        with open(self.LOG_FILE, "a") as f:
            f.write(log_msg + "\n")

        os.makedirs(self.TB_DIR, exist_ok=True)
        writer = SummaryWriter(log_dir=self.TB_DIR)

        # Build extra gym.make kwargs from config (e.g. MuJoCo reward params)
        gym_kwargs = getattr(cfg, "gym_kwargs", None)
        if gym_kwargs is not None:
            gym_kwargs = dict(gym_kwargs)

        # Create parallel envs
        train_envs = make_parallel_envs(
            cfg.dmc_task,
            cfg.action_repeat,
            cfg.size,
            cfg.time_limit,
            cfg.env_num,
            seed=0,
            device=str(device),
            gym_kwargs=gym_kwargs,
        )
        obs_space = train_envs.observation_space
        act_space = train_envs.action_space

        # Buffer
        replay_buffer = Buffer(
            batch_size=cfg.batch_size,
            batch_length=cfg.batch_length,
            max_size=cfg.buffer_max_size,
            device=str(device),
            storage_device="cpu",
        )

        # Resume from checkpoint
        start_step = 0
        update_count = 0
        best_reward = float("-inf")
        rewards_per_episode = []
        ckpt = None
        cfg_runtime = cfg

        if os.path.exists(self.CHECKPOINT_FILE):
            ckpt = torch.load(
                self.CHECKPOINT_FILE, map_location=device, weights_only=False
            )
            cfg_runtime = _with_compat_cfg(cfg, ckpt["agent_state_dict"])

        # Agent
        agent = Dreamer(cfg_runtime, obs_space, act_space).to(device)

        resumed = False
        if ckpt is not None:
            try:
                _load_state_dict_compat(
                    agent,
                    ckpt["agent_state_dict"],
                    "train-resume",
                    min_loaded_ratio=95.0,
                )
                start_step = ckpt.get("step", 0)
                update_count = ckpt.get("update_count", 0)
                best_reward = ckpt.get("best_reward", float("-inf"))
                rewards_per_episode = ckpt.get("rewards_per_episode", [])
                resumed = True
                print(f"Resumed from step {start_step} | best_reward={best_reward:.1f}")
            except RuntimeError as err:
                print(
                    "Checkpoint incompatible with current config, starting from scratch. "
                    f"Details: {err}"
                )
        if not resumed:
            print("Starting training from scratch.")

        # Training loop
        steps = float("inf") if cfg.steps == "inf" else int(cfg.steps)
        action_repeat = int(cfg.action_repeat)
        batch_length = int(cfg.batch_length)

        _should_log = Every(5000)
        _should_pretrain = Once()

        step = start_step
        done = torch.ones(train_envs.env_num, dtype=torch.bool, device=device)
        returns = torch.zeros(train_envs.env_num, dtype=torch.float32, device=device)
        lengths = torch.zeros(train_envs.env_num, dtype=torch.int32, device=device)
        episode_ids = torch.arange(train_envs.env_num, dtype=torch.int32, device=device)
        agent_state = agent.get_initial_state(train_envs.env_num)
        act = agent_state["prev_action"].clone()
        train_metrics = {}
        episode_count = len(rewards_per_episode)
        last_save_time = time.time()
        last_print_time = time.time()
        last_print_step = step

        # Async update state
        _update_thread = None
        _update_result = {}
        _pending_index = None
        _steps_since_update = 0
        _update_denom = int(cfg.batch_size) * batch_length
        _max_updates_per_cycle = int(getattr(cfg, "max_updates_per_cycle", 8))

        print(
            f"Training for {steps} env steps on {cfg.dmc_task}... [device={device}, gpu={torch.cuda.get_device_name(0) if device.type=='cuda' else 'none'}]"
        )

        while step < steps:
            # === Check if background update finished ===
            if _update_thread is not None and not _update_thread.is_alive():
                if _update_result.get("success"):
                    train_metrics = _update_result["metrics"]
                    _pending_index = _update_result.get("index")
                    if (
                        _pending_index is not None
                        and _update_result["stoch"] is not None
                    ):
                        replay_buffer.update(
                            _pending_index,
                            _update_result["stoch"].to(device, non_blocking=True),
                            _update_result["deter"].to(device, non_blocking=True),
                        )
                    update_count += _update_result.get("num_updates", 1)
                elif "error" in _update_result:
                    print(f"\nUpdate error: {_update_result['error']}")
                agent.clone_and_freeze()
                _update_thread = None
                _update_result = {}
                _pending_index = None

                if time.time() - last_save_time > 60:
                    self._save_checkpoint(
                        agent, step, update_count, best_reward, rewards_per_episode
                    )
                    if rewards_per_episode:
                        save_graph(self.GRAPH_FILE, rewards_per_episode)
                    last_save_time = time.time()

            # Log episode completions
            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        ep_reward = returns[i].item()
                        rewards_per_episode.append(ep_reward)
                        episode_count += 1

                        writer.add_scalar("reward/episode", ep_reward, episode_count)
                        mean_100 = np.mean(rewards_per_episode[-100:])
                        writer.add_scalar("reward/mean_100", mean_100, episode_count)

                        if ep_reward > best_reward:
                            best_reward = ep_reward
                            torch.save(agent.state_dict(), self.MODEL_FILE)
                            log_msg = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {ep_reward:.1f} at episode {episode_count}"
                            print(log_msg)
                            with open(self.LOG_FILE, "a") as f:
                                f.write(log_msg + "\n")

                        writer.add_scalar("reward/best", best_reward, episode_count)
                        if episode_count % 20 == 0:
                            writer.flush()

                        if episode_count % 10 == 0:
                            print(
                                f"\nEp {episode_count} | Reward: {ep_reward:.1f} | Mean100: {mean_100:.1f} | Step: {step}"
                            )

                        returns[i] = lengths[i] = 0

            step += int((~done).sum()) * action_repeat
            lengths += ~done

            # Step envs
            act_cpu = act.detach().to("cpu")
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = train_envs.step(act_cpu, done_cpu)
            trans = trans_cpu.to(device, non_blocking=True)
            done = done_cpu.to(device)

            # Policy inference
            act, agent_state = agent.act(trans.clone(), agent_state, eval_mode=False)

            # Store in buffer
            trans["action"] = act * ~done.unsqueeze(-1)
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            trans["episode"] = episode_ids
            replay_buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]

            # === Start async update if ready and no update running ===
            _steps_since_update += int((~done).sum()) * action_repeat
            past_warmup = (
                step // (train_envs.env_num * action_repeat) > batch_length + 1
            )
            buffer_ok = replay_buffer.count() >= int(cfg.batch_size) * (
                batch_length + 1
            )
            if past_warmup and buffer_ok and _update_thread is None:
                if _should_pretrain():
                    pass  # skip first update to let buffer fill a bit more
                else:
                    requested_updates = max(
                        1, int(_steps_since_update * cfg.train_ratio / _update_denom)
                    )
                    num_updates = min(requested_updates, _max_updates_per_cycle)
                    # Consume only the env-steps budget covered by this batch of updates.
                    consumed_steps = int(num_updates * _update_denom / max(1, cfg.train_ratio))
                    _steps_since_update = max(0, _steps_since_update - consumed_steps)
                    _update_result = {}
                    _update_thread = threading.Thread(
                        target=_bg_update_fn,
                        args=(agent, replay_buffer, num_updates, _update_result),
                        daemon=True,
                    )
                    _update_thread.start()

            # Logging
            if _should_log(step) and train_metrics:
                for name, value in train_metrics.items():
                    val = value.item() if isinstance(value, torch.Tensor) else value
                    writer.add_scalar(f"train/{name}", val, step)
                writer.add_scalar("train/updates", update_count, step)
                writer.flush()

            # Periodic checkpoint when idle
            if _update_thread is None and time.time() - last_save_time > 60:
                self._save_checkpoint(
                    agent, step, update_count, best_reward, rewards_per_episode
                )
                if rewards_per_episode:
                    save_graph(self.GRAPH_FILE, rewards_per_episode)
                last_save_time = time.time()

            # Progress line every second
            now = time.time()
            if now - last_print_time >= 1.0:
                elapsed = now - last_print_time
                sps = (step - last_print_step) / elapsed if elapsed > 0 else 0
                mean_100 = (
                    np.mean(rewards_per_episode[-100:]) if rewards_per_episode else 0.0
                )
                last_ep = rewards_per_episode[-1] if rewards_per_episode else 0.0
                print(
                    f"\rEp {episode_count} | Reward: {last_ep:.1f} | Mean100: {mean_100:.1f}"
                    f" | Steps: {step} | Steps/sec: {sps:.1f} | Updates: {update_count}",
                    end="",
                    flush=True,
                )
                last_print_time = now
                last_print_step = step

        # Wait for any in-flight update to finish before final save
        if _update_thread is not None:
            _update_thread.join()
            if _update_result.get("success") and _pending_index is not None:
                replay_buffer.update(
                    _pending_index,
                    _update_result["stoch"].to(device),
                    _update_result["deter"].to(device),
                )
            agent.clone_and_freeze()

        # Final save
        self._save_checkpoint(
            agent, step, update_count, best_reward, rewards_per_episode
        )
        if rewards_per_episode:
            save_graph(self.GRAPH_FILE, rewards_per_episode)

        train_envs.close()
        writer.close()
        print("Training complete.")

    def _save_checkpoint(
        self, agent, step, update_count, best_reward, rewards_per_episode
    ):
        checkpoint = {
            "agent_state_dict": agent.state_dict(),
            "step": step,
            "update_count": update_count,
            "best_reward": best_reward,
            "rewards_per_episode": rewards_per_episode,
        }
        torch.save(checkpoint, self.CHECKPOINT_FILE)
        log_msg = (
            f"{datetime.now().strftime(DATE_FORMAT)}: Checkpoint saved at step {step}"
        )
        print(log_msg)
        with open(self.LOG_FILE, "a") as f:
            f.write(log_msg + "\n")

    # ── Evaluation ─────────────────────────────────────────────────

    def _eval(self, cfg, render):
        os.makedirs(self.EVAL_DIR, exist_ok=True)
        video_file = os.path.join(self.EVAL_DIR, "best.mp4")
        reward_file = os.path.join(self.EVAL_DIR, "best_reward.txt")

        best_eval_reward = float("-inf")
        if os.path.exists(reward_file):
            with open(reward_file, "r") as f:
                best_eval_reward = float(f.read().strip())
            print(f"Previous best eval reward: {best_eval_reward:.1f}")

        # Load model state early to adapt config for legacy checkpoints.
        saved_sd = None
        if os.path.exists(self.MODEL_FILE):
            saved_sd = torch.load(
                self.MODEL_FILE, map_location=device, weights_only=True
            )
            cfg = _with_compat_cfg(cfg, saved_sd)

        # Build extra gym.make kwargs from config
        gym_kwargs = getattr(cfg, "gym_kwargs", None)
        if gym_kwargs is not None:
            gym_kwargs = dict(gym_kwargs)

        # Create single eval env
        env = make_eval_env(
            cfg.dmc_task,
            cfg.action_repeat,
            cfg.size,
            cfg.time_limit,
            seed=42,
            render=render,
            gym_kwargs=gym_kwargs,
        )
        obs_space = env.observation_space
        act_space = env.action_space

        # Create agent and load best model
        agent = Dreamer(cfg, obs_space, act_space).to(device)
        if saved_sd is not None:
            _load_state_dict_compat(agent, saved_sd, "eval")
            print(f"Loaded best model from: {self.MODEL_FILE}")
        else:
            print("WARNING: No best_model.pt found. Running with random weights.")
        agent.eval()

        for episode in itertools.count():
            obs = env.reset()

            state = agent.get_initial_state(1)
            obs_td = _obs_to_td(obs, device)
            episode_reward = 0.0
            done = False

            while not done:
                with torch.no_grad():
                    action, state = agent.act(obs_td, state, eval_mode=True)

                action_np = action.squeeze(0).cpu().numpy()
                obs, reward, done, info = env.step(action_np)
                episode_reward += reward

                obs_td = _obs_to_td(obs, device)

                if render:
                    print(f"Episode Reward: {episode_reward:.1f}", end="\r")

            print(
                f"Episode {episode} | Reward: {episode_reward:.1f} | Best: {best_eval_reward:.1f}"
            )

            if hasattr(env, "save_video") and episode_reward > best_eval_reward:
                best_eval_reward = episode_reward
                env.save_video(video_file)
                with open(reward_file, "w") as f:
                    f.write(f"{best_eval_reward}\n")
                print(f"  -> New best! Video saved to {video_file}")


def main():
    parser = argparse.ArgumentParser(description="R2-Dreamer: Train or evaluate")
    parser.add_argument(
        "hyperparameters", help="Name of the hyperparameter set (e.g. walker_walk)"
    )
    parser.add_argument("--train", help="Training mode", action="store_true")
    args = parser.parse_args()

    agent = R2DreamerAgent(args.hyperparameters)

    if args.train:
        agent.run(is_training=True)
    else:
        agent.run(is_training=False, render=True)


if __name__ == "__main__":
    main()
