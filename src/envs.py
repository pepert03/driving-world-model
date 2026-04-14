import multiprocessing as mp
from functools import partial

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict


# --- DMC Environment Wrapper ---

class DeepMindControl(gym.Env):
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0):
        domain, task = name.rsplit("_", 1)
        from dm_control import suite
        self._env = suite.load(domain, task, task_kwargs={"random": seed})
        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2, fish=3).get(domain, 0)
        self._camera = camera
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            shape = (1,) if len(value.shape) == 0 else value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        obs["is_last"] = time_step.last()
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self, **kwargs):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        obs["is_terminal"] = False
        obs["is_first"] = True
        obs["is_last"] = False
        return obs

    def render(self, *args, **kwargs):
        return self._env.physics.render(*self._size, camera_id=self._camera)


# --- Wrappers ---

class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(np.isfinite(env.action_space.low), np.isfinite(env.action_space.high))
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0, np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self, **kwargs):
        self._step = 0
        return self.env.reset(**kwargs)


class Dtype(gym.Wrapper):
    """Ensure observations have consistent dtypes."""
    def _convert(self, obs):
        for key in obs:
            if isinstance(obs[key], np.ndarray):
                if obs[key].dtype == np.float64:
                    obs[key] = obs[key].astype(np.float32)
            elif isinstance(obs[key], bool):
                obs[key] = np.array(obs[key], dtype=np.bool_)
            elif isinstance(obs[key], (int, float)):
                obs[key] = np.array(obs[key], dtype=np.float32)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._convert(obs), reward, done, info

    def reset(self, **kwargs):
        return self._convert(self.env.reset(**kwargs))


# --- Environment worker for multiprocessing ---

def _worker(conn, constructor):
    import os
    os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "")
    env = constructor()
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            conn.send(("step", obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(("reset", obs))
        elif cmd == "obs_space":
            conn.send(env.observation_space)
        elif cmd == "act_space":
            conn.send(env.action_space)
        elif cmd == "close":
            conn.close()
            break


class ParallelEnv:
    def __init__(self, constructors, device="cpu"):
        self.device = device
        self._parents = []
        self._procs = []
        for constructor in constructors:
            parent, child = mp.Pipe()
            proc = mp.Process(target=_worker, args=(child, constructor), daemon=True)
            proc.start()
            child.close()
            self._parents.append(parent)
            self._procs.append(proc)
        self._parents[0].send(("obs_space", None))
        self._obs_space = self._parents[0].recv()
        self._parents[0].send(("act_space", None))
        self._act_space = self._parents[0].recv()

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._act_space

    @property
    def env_num(self):
        return len(self._parents)

    def step(self, action, done):
        """Step all envs. Reset those that are done."""
        action_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
        done_np = done.detach().cpu().numpy() if isinstance(done, torch.Tensor) else done

        for i, (conn, d) in enumerate(zip(self._parents, done_np)):
            if d:
                conn.send(("reset", None))
            else:
                conn.send(("step", action_np[i]))

        obs_list, rewards, dones = [], [], []
        for i, (conn, d) in enumerate(zip(self._parents, done_np)):
            msg = conn.recv()
            if d:
                _, obs = msg
                obs_list.append(obs)
                rewards.append(0.0)
                dones.append(False)
            else:
                _, obs, r, dn, info = msg
                obs_list.append(obs)
                rewards.append(r)
                dones.append(dn)

        obs_stacked = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0].keys()}
        obs_tensors = {k: torch.as_tensor(v, device="cpu") for k, v in obs_stacked.items()}
        rew = torch.as_tensor(rewards, dtype=torch.float32, device="cpu")

        td = TensorDict({**obs_tensors, "reward": rew}, batch_size=(self.env_num,), device="cpu")
        # Lift scalar dims
        for key in td.keys():
            if td[key].ndim == 1:
                td[key] = td[key].unsqueeze(-1)
        td = td.pin_memory()
        done_t = torch.as_tensor(dones, device="cpu")
        return td, done_t

    def close(self):
        for conn in self._parents:
            try:
                conn.send(("close", None))
                conn.close()
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=5)


# --- Factory ---

def make_env(task, action_repeat, size, time_limit, seed):
    _, name = task.split("_", 1)
    env = DeepMindControl(name, action_repeat, tuple(size), seed=seed)
    env = NormalizeActions(env)
    env = TimeLimit(env, time_limit // action_repeat)
    env = Dtype(env)
    return env


def make_envs(config, device):
    time_limit = int(config.time_limit)
    constructors_train = [
        partial(make_env, config.task, config.action_repeat, config.size, time_limit, config.seed + i)
        for i in range(config.env_num)
    ]
    constructors_eval = [
        partial(make_env, config.task, config.action_repeat, config.size, time_limit, config.seed + 1000 + i)
        for i in range(config.eval_episode_num)
    ]
    train_envs = ParallelEnv(constructors_train, device)
    eval_envs = ParallelEnv(constructors_eval, device)
    return train_envs, eval_envs
