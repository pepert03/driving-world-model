from functools import partial

import cv2
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

    def render_high_res(self, size=480):
        return self._env.physics.render(size, size, camera_id=self._camera)


# --- Eval wrapper with display + video recording ---

class EvalRenderWrapper:
    """Wraps a DMC env for eval: displays frames via cv2 and records for video."""

    def __init__(self, env, display_size=480, window_name="R2-Dreamer Eval"):
        self.env = env
        self._display_size = display_size
        self._window_name = window_name
        self._episode_frames = []

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, **kwargs):
        self._episode_frames = []
        obs = self.env.reset(**kwargs)
        self._capture_frame()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._capture_frame()
        return obs, reward, done, info

    def _capture_frame(self):
        frame = self.env.render_high_res(self._display_size)
        self._episode_frames.append(frame.copy())
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(self._window_name, frame_bgr)
        cv2.waitKey(1)

    def save_video(self, path, fps=30):
        if not self._episode_frames:
            return
        import imageio.v3 as iio
        iio.imwrite(path, self._episode_frames, fps=fps, codec="libx264")

    def close(self):
        cv2.destroyAllWindows()
        self.env.close() if hasattr(self.env, 'close') else None


# --- Wrappers ---

class NormalizeActions:
    def __init__(self, env):
        self.env = env
        spec = env.action_space
        self._mask = np.logical_and(np.isfinite(spec.low), np.isfinite(spec.high))
        self._low = np.where(self._mask, spec.low, -1)
        self._high = np.where(self._mask, spec.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    @property
    def observation_space(self):
        return self.env.observation_space

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def close(self):
        return self.env.close() if hasattr(self.env, 'close') else None


class TimeLimit:
    def __init__(self, env, duration):
        self.env = env
        self._duration = duration
        self._step = None

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

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

    def close(self):
        return self.env.close() if hasattr(self.env, 'close') else None


class Dtype:
    def __init__(self, env):
        self.env = env

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

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

    def close(self):
        return self.env.close() if hasattr(self.env, 'close') else None


# --- Vector env (sequential, runs in main process) ---

class VectorEnv:
    """Run multiple envs sequentially in the main process.

    This avoids Windows multiprocessing + OpenGL context issues with dm_control.
    """

    def __init__(self, constructors, device="cpu"):
        self.device = device
        self._envs = [c() for c in constructors]
        self._obs_space = self._envs[0].observation_space
        self._act_space = self._envs[0].action_space

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._act_space

    @property
    def env_num(self):
        return len(self._envs)

    def step(self, action, done):
        action_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
        done_np = done.detach().cpu().numpy() if isinstance(done, torch.Tensor) else done

        obs_list, rewards, dones = [], [], []
        for i, (env, d) in enumerate(zip(self._envs, done_np)):
            if d:
                obs = env.reset()
                obs_list.append(obs)
                rewards.append(0.0)
                dones.append(False)
            else:
                obs, r, dn, info = env.step(action_np[i])
                obs_list.append(obs)
                rewards.append(r)
                dones.append(dn)

        obs_stacked = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0].keys()}
        obs_tensors = {k: torch.as_tensor(v, device="cpu") for k, v in obs_stacked.items()}
        rew = torch.as_tensor(rewards, dtype=torch.float32, device="cpu")
        td = TensorDict({**obs_tensors, "reward": rew}, batch_size=(self.env_num,), device="cpu")
        for key in td.keys():
            if td[key].ndim == 1:
                td[key] = td[key].unsqueeze(-1)
        done_t = torch.as_tensor(dones, device="cpu")
        return td, done_t

    def close(self):
        for env in self._envs:
            if hasattr(env, 'close'):
                env.close()


# --- Factories ---

def make_env(dmc_task, action_repeat, size, time_limit, seed):
    env = DeepMindControl(dmc_task, action_repeat, tuple(size), seed=seed)
    env = NormalizeActions(env)
    env = TimeLimit(env, time_limit // action_repeat)
    env = Dtype(env)
    return env


def make_eval_env(dmc_task, action_repeat, size, time_limit, seed=42, render=False):
    """Create a single env for eval, optionally with display + video recording."""
    env = DeepMindControl(dmc_task, action_repeat, tuple(size), seed=seed)
    env = NormalizeActions(env)
    env = TimeLimit(env, time_limit // action_repeat)
    env = Dtype(env)
    if render:
        env = EvalRenderWrapper(env)
    return env


def make_parallel_envs(dmc_task, action_repeat, size, time_limit, env_num, seed, device):
    constructors = [
        partial(make_env, dmc_task, action_repeat, size, time_limit, seed + i)
        for i in range(env_num)
    ]
    return VectorEnv(constructors, device)
