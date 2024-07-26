import os
import pickle
import gzip
import numpy as np
import spatialmath as sm
from typing import List 
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EpisodeDemoData:
    l_marker_flow: np.ndarray
    r_marker_flow: np.ndarray
    ee_poses: np.ndarray
    actions: np.ndarray
    # For debugging purposes
    ee_init_world_pose: np.ndarray
    actions_relative: np.ndarray
    max_action_relative: np.ndarray


def stack_markers(markers:np.ndarray):
    # markers have shape: 2x128x2 ((original, displaced) x num_of_markers x (x, y))
    # Output shape: 128x4
    markers_reshaped = np.concatenate([markers[0], markers[1]], axis=-1)
    return markers_reshaped


def normalize_and_store_data(data: List[EpisodeDemoData], file_name, datasets_dir="./data"):
    # Convert list of EpisodeDemoData to dictionary for easier training
    l_marker_flow = np.concatenate([data_episode.l_marker_flow for data_episode in data], axis=0, dtype=np.float32)
    r_marker_flow = np.concatenate([data_episode.r_marker_flow for data_episode in data], axis=0, dtype=np.float32)
    ee_poses = np.concatenate([data_episode.ee_poses for data_episode in data], axis=0, dtype=np.float32)
    actions = np.concatenate([data_episode.actions for data_episode in data], axis=0, dtype=np.float32)

    # Normalize and create stats here
    normalization_stats = dict()
    normalization_stats["ee_poses"] = get_data_stats(ee_poses)
    normalization_stats["actions"] = get_data_stats(actions)
    ee_poses_normalized = normalize_data(ee_poses, normalization_stats["ee_poses"])
    actions_normalized = normalize_data(actions, normalization_stats["actions"])

    episode_ends = np.cumsum([data_episode.actions.shape[0] for data_episode in data], dtype=np.int64)

    # Save for debugging
    ee_initial_world_pose = np.stack([data_episode.ee_init_world_pose for data_episode in data], axis=0, dtype=np.float32)
    actions_relative = np.concatenate([data_episode.actions_relative for data_episode in data], axis=0, dtype=np.float32)
    max_action_relative = np.stack([data_episode.max_action_relative for data_episode in data], axis=0, dtype=np.float32)

    data = {
        "data": {
            "l_marker_flow": l_marker_flow,
            "r_marker_flow": r_marker_flow,
            "ee_poses": ee_poses_normalized,
            "actions": actions_normalized,
        },
        "meta": {
            "episode_ends": episode_ends,
            "normalization_stats": normalization_stats,
            "max_action_relative": max_action_relative
        },
        "debug": {
            # Save for debugging
            "ee_initial_world_pose": ee_initial_world_pose,
            "actions_relative": actions_relative
        }
    }

    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, f"{file_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pkl.gzip")
    f = gzip.open(data_file, "wb")
    pickle.dump(data, f)
    f.close()


def action_percentage_to_value(action:np.ndarray, max_action:np.ndarray):
    # action in range (-1, 1), shape (, action_dim)
    # max_action has same shape as action
    return action * max_action


def action_value_to_percentage(action:np.ndarray, max_action:np.ndarray):
    # action in range (-max_action, max_action), shape (, action_dim)
    # max_action has same shape as action
    return action / max_action


def transformation_matrix_to_xyz_rpy(transform):
    transform = sm.SE3.Rt(transform[0:3, 0:3], transform[0:3, -1], check=False)
    xyz = transform.t
    rpy = transform.rpy()
    return xyz, rpy


def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data
