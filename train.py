import pathlib
import sys
import warnings

import hydra
import torch
from omegaconf import OmegaConf

from src.agent import Dreamer
from src.buffer import Buffer
from src.envs import make_envs
from src.tools import Logger, Every, Once, set_seed_everywhere, to_np

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    set_seed_everywhere(config.seed)
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    print(f"Logdir: {logdir}")
    print(f"Device: {device}")
    print(f"Task: {config.env.task}")

    logger = Logger(logdir)
    logger.log_hydra_config(config)

    # Buffer
    replay_buffer = Buffer(
        batch_size=config.training.batch_size,
        batch_length=config.training.batch_length,
        max_size=config.buffer.max_size,
        device=config.device,
        storage_device=config.buffer.storage_device,
    )

    # Environments
    print("Creating environments...")
    train_envs, eval_envs = make_envs(config.env, config.device)
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    print(f"Obs space: {obs_space}")
    print(f"Act space: {act_space}")

    # Agent
    print("Creating agent...")
    agent = Dreamer(config.model, obs_space, act_space).to(device)

    # Training helpers
    steps = int(config.training.steps)
    eval_every = int(config.training.eval_every)
    log_every = int(config.training.log_every)
    batch_length = int(config.training.batch_length)
    action_repeat = int(config.env.action_repeat)
    batch_steps = int(config.training.batch_size * config.training.batch_length)
    train_ratio = int(config.training.train_ratio)

    _should_eval = Every(eval_every)
    _should_log = Every(log_every)
    _should_pretrain = Once()
    _updates_needed = Every(batch_steps / train_ratio * action_repeat)

    # Training state
    step = 0
    update_count = 0
    done = torch.ones(train_envs.env_num, dtype=torch.bool, device=device)
    returns = torch.zeros(train_envs.env_num, dtype=torch.float32, device=device)
    lengths = torch.zeros(train_envs.env_num, dtype=torch.int32, device=device)
    episode_ids = torch.arange(train_envs.env_num, dtype=torch.int32, device=device)
    agent_state = agent.get_initial_state(train_envs.env_num)
    act = agent_state["prev_action"].clone()
    train_metrics = {}

    print(f"Starting training for {steps} steps...")

    while step < steps:
        # Evaluation
        if _should_eval(step) and config.env.eval_episode_num > 0:
            eval_agent(agent, eval_envs, logger, step, batch_length)

        # Log episode completions
        if done.any():
            for i, d in enumerate(done):
                if d and lengths[i] > 0:
                    logger.scalar("episode/score", returns[i])
                    logger.scalar("episode/length", lengths[i])
                    logger.write(step + i)
                    returns[i] = lengths[i] = 0

        step += int((~done).sum()) * action_repeat
        lengths += ~done

        # Step environments (CPU side)
        act_cpu = act.detach().to("cpu")
        done_cpu = done.detach().to("cpu")
        trans_cpu, done_cpu = train_envs.step(act_cpu, done_cpu)

        # Transfer to GPU
        trans = trans_cpu.to(device, non_blocking=True)
        done = done_cpu.to(device)

        # Policy inference
        act, agent_state = agent.act(trans.clone(), agent_state, eval_mode=False)

        # Store transition
        trans["action"] = act * ~done.unsqueeze(-1)
        trans["stoch"] = agent_state["stoch"]
        trans["deter"] = agent_state["deter"]
        trans["episode"] = episode_ids
        replay_buffer.add_transition(trans.detach())
        returns += trans["reward"][:, 0]

        # Update model
        if step // (train_envs.env_num * action_repeat) > batch_length + 1:
            if _should_pretrain():
                update_num = int(config.training.pretrain) if config.training.pretrain else 0
            else:
                update_num = _updates_needed(step)
            for _ in range(update_num):
                train_metrics = agent.update(replay_buffer)
            update_count += update_num

            if _should_log(step) and train_metrics:
                for name, value in train_metrics.items():
                    val = value.detach().cpu().item() if isinstance(value, torch.Tensor) else value
                    logger.scalar(f"train/{name}", val)
                logger.scalar("train/updates", update_count)
                logger.write(step, fps=True)

    # Save final checkpoint
    print("Saving checkpoint...")
    torch.save({
        "agent_state_dict": agent.state_dict(),
    }, logdir / "latest.pt")

    train_envs.close()
    eval_envs.close()
    print("Training complete.")


@torch.no_grad()
def eval_agent(agent, envs, logger, train_step, batch_length):
    print("Evaluating...")
    agent.eval()
    done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
    once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=agent.device)
    returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
    steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
    agent_state = agent.get_initial_state(envs.env_num)
    act = agent_state["prev_action"].clone()

    while not once_done.all():
        steps += ~done * ~once_done
        act_cpu = act.detach().to("cpu")
        done_cpu = done.detach().to("cpu")
        trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
        trans = trans_cpu.to(agent.device, non_blocking=True)
        done = done_cpu.to(agent.device)
        trans["action"] = act
        act, agent_state = agent.act(trans, agent_state, eval_mode=True)
        returns += trans["reward"][:, 0] * ~once_done
        once_done |= done

    logger.scalar("episode/eval_score", returns.mean())
    logger.scalar("episode/eval_length", steps.to(torch.float32).mean())
    logger.write(train_step)
    agent.train()
    print(f"  Eval score: {returns.mean().item():.1f}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
