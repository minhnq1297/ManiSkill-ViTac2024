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
        n_obs_steps: int,
        n_pred_steps: int,
        subs_factor: int = 1,  # 1 means no subsampling
        **kwargs,
    ) -> None:
        replay_buffer = SimpleReplayBuffer.create_from_path(data_path)
        data_keys = ["l_marker_flow", "r_marker_flow", "ee_poses", "actions"]
        data_key_first_k = {
            "l_marker_flow": n_obs_steps * subs_factor,
            "r_marker_flow": n_obs_steps * subs_factor,
            "ee_poses": n_obs_steps * subs_factor
        }
        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=(n_obs_steps + n_pred_steps) * subs_factor - (subs_factor - 1),
            pad_before=(n_obs_steps - 1) * subs_factor,
            pad_after=(n_pred_steps - 1) * subs_factor + (subs_factor - 1),
            keys=data_keys,
            key_first_k=data_key_first_k,
        )
        self.normalization_stats = replay_buffer.meta["normalization_stats"]
        self.n_obs_steps = n_obs_steps
        self.n_prediction_steps = n_pred_steps
        self.subs_factor = subs_factor
        self.rng = np.random.default_rng()
        return

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        sample: dict[str, np.ndarray] = self.sampler.sample_sequence(idx)
        cur_step_i = self.n_obs_steps * self.subs_factor
        l_marker_flow = sample["l_marker_flow"][: cur_step_i : self.subs_factor]
        r_marker_flow = sample["r_marker_flow"][: cur_step_i : self.subs_factor]
        ee_poses = sample["ee_poses"][: cur_step_i : self.subs_factor]
        actions = sample["actions"][cur_step_i :: self.subs_factor]
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
