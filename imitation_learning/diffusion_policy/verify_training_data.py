import os
import sys
import numpy as np
import torch
import torch.nn as nn

script_path = os.path.dirname(os.path.realpath(__file__))
repo_path = os.path.join(script_path, "../..")
sys.path.append(script_path)
sys.path.insert(0, repo_path)

from imitation_learning.diffusion_policy.diffusion_policy_dataset import DiffusionPolicyDataset
from imitation_learning.utils import *

dataset_path = "./data/open_lock_demo-20240726-012658.pkl.gzip"
obs_horizon = 4
pred_horizon = 4
action_horizon = 1

train_dataset = DiffusionPolicyDataset(
    data_path=dataset_path,
    obs_horizon=obs_horizon,
    pred_horizon=pred_horizon,
    action_horizon=action_horizon
)

test_data = train_dataset[3]

normalization_stats = train_dataset.get_normalization_stats()
normalization_ee_poses_stats = normalization_stats["ee_poses"]
normalization_actions_stats = normalization_stats["actions"]

test_ee_poses = unnormalize_data(test_data["ee_poses"].numpy(), normalization_ee_poses_stats)
test_actions = unnormalize_data(test_data["actions"].numpy(), normalization_actions_stats)
print(test_ee_poses)
print(test_actions)

print("################## Load data to cross check #################")

# Load expert data to cross check if it's padded or not?
data_path = "./data/open_lock_demo-20240726-012658.pkl.gzip"
f = gzip.open(data_path, "rb")
raw_dataset = pickle.load(f)

episode_ends = raw_dataset["meta"]["episode_ends"]
episode_start_ind = 0
episode_end_ind = episode_ends[0]

data_ee_poses = raw_dataset["data"]["ee_poses"][episode_start_ind:episode_end_ind]
data_ee_poses = unnormalize_data(data_ee_poses, normalization_ee_poses_stats)
print(data_ee_poses)

data_actions = raw_dataset["data"]["actions"][episode_start_ind:episode_end_ind]
data_actions = unnormalize_data(data_actions, normalization_actions_stats)
print(data_actions)


