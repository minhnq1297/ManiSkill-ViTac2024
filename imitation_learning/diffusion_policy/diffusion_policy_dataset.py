import pickle
import gzip
import torch
import numpy as np
from typing import Dict
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.common.pytorch_util import dict_apply


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(self, root: Dict[str, dict]):
        assert('data' in root)
        assert('meta' in root)
        assert('episode_ends' in root['meta'])
        for key, value in root['data'].items():
            assert(value.shape[0] == root['meta']['episode_ends'][-1])
        self.root = root

    @classmethod
    def create_from_path(cls, data_path):
        f = gzip.open(data_path, "rb")
        dict = pickle.load(f)
        return cls(root=dict)


class DiffusionPolicyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        obs_horizon: int,
        pred_horizon: int,
        action_horizon: int,
    ) -> None:
        replay_buffer = SimpleReplayBuffer.create_from_path(data_path)
        data_keys = ["l_marker_flow", "r_marker_flow", "ee_poses", "actions"]
        data_key_first_k = {
            "l_marker_flow": obs_horizon,
            "r_marker_flow": obs_horizon,
            "ee_poses": obs_horizon,
        }
        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
            keys=data_keys,
            key_first_k=data_key_first_k,
        )
        self.normalization_stats = replay_buffer.meta["normalization_stats"]
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.rng = np.random.default_rng()
        return

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        sample: dict[str, np.ndarray] = self.sampler.sample_sequence(idx)
        l_marker_flow = sample["l_marker_flow"][:self.obs_horizon, :]
        r_marker_flow = sample["r_marker_flow"][:self.obs_horizon, :]
        ee_poses = sample["ee_poses"][:self.obs_horizon, :]
        actions = sample["actions"]
        sample = {
            "l_marker_flow": l_marker_flow,
            "r_marker_flow": r_marker_flow,
            "ee_poses": ee_poses,
            "actions": actions,
        }
        torch_data = dict_apply(sample, torch.from_numpy)
        return torch_data

    def get_normalization_stats(self):
        return self.normalization_stats
