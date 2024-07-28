import os
import sys
import copy
import collections
import torch
import argparse
import numpy as np
import torch.nn as nn
import ruamel.yaml as yaml
import spatialmath as sm

from path import Path
from loguru import logger
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from stable_baselines3.common.utils import set_random_seed

script_path = os.path.dirname(os.path.realpath(__file__))
repo_path = os.path.join(script_path, "../..")
sys.path.append(script_path)
sys.path.insert(0, repo_path)

from solutions.networks import PointNetFeatureExtractor
from utils.common import get_time, get_average_params
from imitation_learning.utils import *

DEVICE = torch.device('cuda')

model_path = "./trained_model/checkpoint_open_lock_model_20240726-013001.pth.tar"
use_pretrained_encoder = True

# Dimensions
obs_horizon = 4
action_horizon = 2
pred_horizon = 4

vision_feature_dim = 64
robot_pose_dim = 3
obs_dim = vision_feature_dim + robot_pose_dim
action_dim = 3
# Network setup
noise_pred_net = ConditionalUnet1D(input_dim=action_dim,
                                global_cond_dim=obs_dim*obs_horizon, cond_predict_scale=True)
vision_encoder = PointNetFeatureExtractor(dim=4, out_dim=32)
# Load checkpoint
check_point = torch.load(model_path)
# Load and freeze trained Diffusion network
noise_pred_net_state_dict = check_point["noise_pred_net_statedict"]
noise_pred_net.load_state_dict(noise_pred_net_state_dict)
noise_pred_net.eval()

# Load and freeze pretrained encoder
if use_pretrained_encoder:
    vision_encoder_state_dict = torch.load("./pretrain_weight/pretrain_openlock/marker_encoder_openlock.zip")
else:
    vision_encoder_state_dict = check_point["visual_encoder_statedict"]
vision_encoder.load_state_dict(vision_encoder_state_dict)
vision_encoder.eval()

# Load normalization dictionary
normalization_stats = check_point["normalization_stats"]
normalization_ee_poses_stats = normalization_stats["ee_poses"]
normalization_actions_stats = normalization_stats["actions"]

print("Agent Normalization Stats")
print(normalization_ee_poses_stats)
print(normalization_actions_stats)

nets = nn.ModuleDict({
    'visual_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
})

nets.to(DEVICE)

# Get noise scheduler
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,  #NOTE: Change this to False if we do not need to clip the output to [-1,1]
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)
noise_scheduler.set_timesteps(num_diffusion_iters)

# Load expert data
data_path = "./data/open_lock_demo-20240726-012658.pkl.gzip"
f = gzip.open(data_path, "rb")
dataset = pickle.load(f)

data_normalization_stats = dataset["meta"]["normalization_stats"]
data_normalization_ee_poses_stats = data_normalization_stats["ee_poses"]
data_normalization_actions_stats = data_normalization_stats["actions"]
print("Data Normalization Stats")
print(data_normalization_ee_poses_stats)
print(data_normalization_actions_stats)

# Choose 1 episode
max_action = dataset["meta"]["max_action_relative"]

episode_ends = dataset["meta"]["episode_ends"]
episode_start_ind = episode_ends[0]
episode_end_ind = episode_ends[1]

# Take the data out and padding
def padding_data(data, padding_length):
    padding_element = np.expand_dims(data[0], axis=0)
    padding = np.repeat(padding_element, padding_length, axis=0)
    padded_data = np.concatenate([padding, data], axis=0)
    return padded_data

n_padding = obs_horizon - 1
full_ee_poses = dataset["data"]["ee_poses"][episode_start_ind:episode_end_ind]
full_ee_poses = padding_data(full_ee_poses, n_padding)
full_ee_poses = torch.from_numpy(full_ee_poses).to(device=DEVICE, dtype=torch.float32)

full_actions = dataset["data"]["actions"][episode_start_ind:episode_end_ind]

full_l_marker_flow = dataset["data"]["l_marker_flow"][episode_start_ind:episode_end_ind]
full_l_marker_flow = padding_data(full_l_marker_flow, n_padding)

full_r_marker_flow = dataset["data"]["r_marker_flow"][episode_start_ind:episode_end_ind]
full_r_marker_flow = padding_data(full_r_marker_flow, n_padding)

# Slice to get obs_horizon
# start_ind >= episode_start_ind and (start_ind + obs_horizon) < episode_end_ind
action_execute_list = []
print(episode_start_ind, episode_end_ind)
for i in range(episode_end_ind - episode_start_ind):
    start_ind = i
    ee_poses = full_ee_poses[start_ind:start_ind+obs_horizon]

    l_marker_flow = full_l_marker_flow[start_ind:start_ind+obs_horizon]
    r_marker_flow = full_r_marker_flow[start_ind:start_ind+obs_horizon]
    marker_flow = np.concatenate([l_marker_flow, r_marker_flow], axis=0)
    marker_flow = torch.from_numpy(marker_flow).to(device=DEVICE, dtype=torch.float32)

    obs_features = vision_encoder(marker_flow)
    obs_features = torch.cat([obs_features[:obs_horizon], obs_features[obs_horizon:]], dim=-1)
    obs_features = torch.cat([obs_features, ee_poses], dim=-1)
    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

    # Denoising to get output
    noisy_action = torch.randn(
        (1, pred_horizon, action_dim), device=DEVICE)
    naction = noisy_action


    for k in noise_scheduler.timesteps:
        # predict noise
        noise_pred = nets["noise_pred_net"](
            sample=naction,
            timestep=k,
            global_cond=obs_cond
        )

        # inverse diffusion step (remove noise)
        naction = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=naction
        ).prev_sample


    naction = naction.cpu().detach().numpy()
    naction = naction[0]

    action_pred = unnormalize_data(naction, stats=normalization_actions_stats)
    action_execute_list.append(action_pred[0])

action_execute = np.array(action_execute_list)

unnormalized_actions = unnormalize_data(full_actions, stats=normalization_actions_stats)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

ax.scatter(action_execute[:, 0], action_execute[:, 1], action_execute[:, 2], color="r", label="agent_action")
ax.scatter(unnormalized_actions[:, 0], unnormalized_actions[:, 1], unnormalized_actions[:, 2], color="b", label="expert_action")
ax.legend()

for i in range(action_execute.shape[0]):
    ax.text(action_execute[i, 0], action_execute[i, 1], action_execute[i, 2], str(i))
for i in range(unnormalized_actions.shape[0]):
    ax.text(unnormalized_actions[i, 0], unnormalized_actions[i, 1], unnormalized_actions[i, 2], str(i))

plt.show()
