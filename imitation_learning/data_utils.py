import os
import pickle
import gzip
import torch
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.common.pytorch_util import dict_apply

@dataclass
class EpisodeDemoData:
    l_marker_flow: np.ndarray
    r_marker_flow: np.ndarray
    actions: np.ndarray
    # ee_pose: List[np.ndarray]
    # done_flag: List[bool]


def stack_markers(markers:np.ndarray):
    # markers have shape: 2x128x2 ((original, displaced) x num_of_markers x (x, y))
    # Output shape: 128x4
    markers_reshaped = np.concatenate([markers[0], markers[1]], axis=-1)
    return markers_reshaped


def store_data(data: List[EpisodeDemoData], file_name, datasets_dir="./data"):
    # Convert list of EpisodeDemoData to dictionary for easier training
    l_marker_flow = np.concatenate([data_episode.l_marker_flow for data_episode in data], axis=0, dtype=np.float32)
    r_marker_flow = np.concatenate([data_episode.r_marker_flow for data_episode in data], axis=0, dtype=np.float32)
    actions = np.concatenate([data_episode.actions for data_episode in data], axis=0, dtype=np.float32)
    episode_ends = np.cumsum([data_episode.actions.shape[0] for data_episode in data], dtype=np.int64)
    data = {
        "data": {
            "l_marker_flow": l_marker_flow,
            "r_marker_flow": r_marker_flow,
            "actions": actions
        },
        "meta": {
            "episode_ends": episode_ends
        }
    }
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, f"{file_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pkl.gzip")
    f = gzip.open(data_file, "wb")
    pickle.dump(data, f)


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(self, root: Dict[str, dict]):
        assert('data' in root)
        assert('meta' in root)
        assert('episode_ends' in root['meta'])
        for key, value in root['data'].items():
            assert(value.shape[0] == root['meta']['episode_ends'][-1])
        self.root = root
        self.root = root

    @classmethod
    def create_from_path(cls, file_name, datasets_dir="./data"):
        data_file = os.path.join(datasets_dir, file_name)
        f = gzip.open(data_file, "rb")
        dict = pickle.load(f)
        return cls(root=dict)


class ViTacDemoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        n_obs_steps: int,
        n_pred_steps: int,
        subs_factor: int = 1,  # 1 means no subsampling
        **kwargs,
    ) -> None:
        replay_buffer = SimpleReplayBuffer.create_from_path(data_path)
        data_keys = ["l_marker_flow", "r_marker_flow", "actions"]
        data_key_first_k = {
            "l_marker_flow": n_obs_steps * subs_factor,
            "r_marker_flow": n_obs_steps * subs_factor,
        }
        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=(n_obs_steps + n_pred_steps) * subs_factor - (subs_factor - 1),
            pad_before=(n_obs_steps - 1) * subs_factor,
            pad_after=(n_pred_steps - 1) * subs_factor + (subs_factor - 1),
            keys=data_keys,
            key_first_k=data_key_first_k,
        )
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
        actions = sample["actions"][: cur_step_i : self.subs_factor]
        sample = {
            "l_marker_flow": l_marker_flow,
            "r_marker_flow": r_marker_flow,
            "actions": actions
        }
        torch_data = dict_apply(sample, torch.from_numpy)
        return torch_data

