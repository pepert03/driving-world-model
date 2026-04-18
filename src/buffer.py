import torch
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer:
    def __init__(self, batch_size, batch_length, max_size, device, storage_device):
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device)
        self.batch_size = int(batch_size)
        self.batch_length = int(batch_length)
        self._buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=int(max_size), device=self.storage_device, ndim=2),
            sampler=SliceSampler(
                num_slices=self.batch_size,
                end_key=None,
                traj_key="episode",
                truncated_key=None,
                strict_length=True,
            ),
            prefetch=0,
            batch_size=self.batch_size * (self.batch_length + 1),
        )

    def add_transition(self, data):
        self._buffer.extend(data.unsqueeze(1))

    def sample(self):
        sample_td, info = self._buffer.sample(return_info=True)
        sample_td = sample_td.view(-1, self.batch_length + 1)
        src_dev = sample_td.device
        if src_dev.type == "cpu" and self.device.type == "cuda":
            sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
        elif src_dev != self.device:
            sample_td = sample_td.to(self.device, non_blocking=True)
        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        data = sample_td[:, 1:]
        data.set_("action", sample_td["action"][:, :-1])
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in info["index"]]
        return data, index, initial

    def update(self, index, stoch, deter):
        index = [ind.reshape(-1) for ind in index]
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        deter = deter.reshape(-1, *deter.shape[2:])
        # Storage is ndim=2: (length, env_num). index[0]=env, index[1]=step.
        self._buffer[index[1], index[0]].set_("stoch", stoch)
        self._buffer[index[1], index[0]].set_("deter", deter)

    def count(self):
        return len(self._buffer)
