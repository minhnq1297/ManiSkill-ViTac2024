import copy
import os
import sys
import argparse
import numpy as np
import ruamel.yaml as yaml
from loguru import logger

script_path = os.path.dirname(os.path.realpath(__file__))
repo_path = os.path.join(script_path, "../..")
sys.path.append(script_path)
sys.path.insert(0, repo_path)

from scripts.arguments import parse_params, handle_policy_args
from envs.long_open_lock import LongOpenLockRandPointFlowEnv
from path import Path
from stable_baselines3.common.utils import set_random_seed
from utils.common import get_time, get_average_params
from loguru import logger

from imitation_learning.data_generation.open_lock_simple_agent import OpenLockSimpleAgent
from imitation_learning.utils import *

import matplotlib.pyplot as plt

EVAL_CFG_FILE = os.path.join(repo_path, "configs/parameters/long_open_lock_demo_gen.yaml")
KEY_NUM = 3
REPEAT_NUM = 1

def demo_generation(model, num_of_offsets):
    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}", level="INFO")

    with open(EVAL_CFG_FILE, "r") as f:
        cfg = yaml.YAML(typ='safe', pure=True).load(f)

    # get simulation and environment parameters
    sim_params = cfg["env"].pop("params")
    env_name = cfg["env"].pop("env_name")

    params_lb, params_ub = parse_params(env_name, sim_params)
    average_params = get_average_params(params_lb, params_ub)
    logger.info(f"\n{average_params}")
    logger.info(cfg["env"])

    if "max_action" in cfg["env"].keys():
        cfg["env"]["max_action"] = np.array(cfg["env"]["max_action"])
        max_action = cfg["env"]["max_action"]
        model.set_max_action(max_action)

    specified_env_args = copy.deepcopy(cfg["env"])

    specified_env_args.update(
        {
            "params": average_params,
            "params_upper_bound": average_params,
        }
    )

    env = LongOpenLockRandPointFlowEnv(**specified_env_args)
    set_random_seed(0)

    max_key_x_offset = cfg["env"]["key_x_max_offset"]
    max_key_y_offset = cfg["env"]["key_y_max_offset"]
    max_key_z_offset = cfg["env"]["key_z_max_offset"]
    max_key_offset = np.array([max_key_x_offset, max_key_y_offset, max_key_z_offset])
    offset_list = max_key_offset * np.random.rand(num_of_offsets, 3)
    offset_list = offset_list.tolist()

    episode_num = len(offset_list)
    collect_result = []

    episode_demo_data_list = []
    for i in range(KEY_NUM):
        for r in range(REPEAT_NUM):
            for k in range(episode_num):
                logger.opt(colors=True).info(f"<blue>#### Run No. {len(collect_result) + 1} ####</blue>")
                o, _ = env.reset(offset_list[k], key_idx=i)

                l_marker_list = []
                r_marker_list = []
                key_transform_list = []
                action_list = []

                key_transform = o["key_transform"]
                key_transform_list.append(key_transform)
                marker_pos = o["marker_flow"]
                l_marker_pos, r_marker_pos = marker_pos[0], marker_pos[1]
                l_marker_list.append(stack_markers(l_marker_pos))
                r_marker_list.append(stack_markers(r_marker_pos))

                d, ep_ret, ep_len = False, 0, 0
                while not d:
                    ep_len += 1
                    action = model.predict(o)
                    action_list.append(action)
                    logger.info(f"Step {ep_len} Action: {action}")
                    o, r, terminated, truncated, info = env.step(action)
                    d = terminated or truncated
                    ep_ret += r

                    # obs_sub_steps = o["obs_sub_steps"]
                    # for obs_sub_step in obs_sub_steps:
                    #     action_sub_step = model.predict(obs_sub_step)
                    #     action_list.append(action_sub_step)
                    #     marker_pos_sub_step = obs_sub_step["marker_flow"]
                    #     l_marker_pos_sub_step, r_marker_pos_sub_step = marker_pos_sub_step[0], marker_pos_sub_step[1]
                    #     l_marker_list.append(stack_markers(l_marker_pos_sub_step))
                    #     r_marker_list.append(stack_markers(r_marker_pos_sub_step))

                    key_transform = o["key_transform"]
                    key_transform_list.append(key_transform)
                    marker_pos = o["marker_flow"]
                    l_marker_pos, r_marker_pos = marker_pos[0], marker_pos[1]
                    l_marker_list.append(stack_markers(l_marker_pos))
                    r_marker_list.append(stack_markers(r_marker_pos))

                if info["is_success"]:
                    collect_result.append([True, ep_len])
                    logger.opt(colors=True).info(f"<green>RESULT: SUCCESS</green>")
                    action_list.append(np.zeros_like(action))
                    episode_demo_data = preprocessing_episode_data(l_marker_list, r_marker_list,
                                                                   key_transform_list, action_list, max_action)
                    episode_demo_data_list.append(episode_demo_data)
                else:
                    collect_result.append([False, ep_len])
                    logger.opt(colors=True).info(f"<d>RESULT: FAIL</d>")

    normalize_and_store_data(episode_demo_data_list, "open_lock_demo")

    env.close()
    success_rate = np.sum(np.array([int(v[0]) for v in collect_result])) / (episode_num * KEY_NUM * REPEAT_NUM)
    if success_rate > 0:
        avg_steps = np.mean(np.array([int(v[1]) if v[0] else 0 for v in collect_result])) / success_rate
        logger.info(f"#SUCCESS_RATE: {success_rate*100.0:.2f}%")
        logger.info(f"#AVG_STEP: {avg_steps:.2f}")
    else:
        logger.info(f"#SUCCESS_RATE: 0")
        logger.info(f"#AVG_STEP: NA")


def preprocessing_episode_data(l_marker_list, r_marker_list, key_transform_list, action_list, max_action):
    # Every data is collected in m. Due to normalization, this does not effect the training
    # Convert action back to normal value in m
    action_value_list = [action_percentage_to_value(action, max_action) * 1e-3 for action in action_list]

    # We always consider the first transform of the key as the origin for others
    # Convert transform w.r.t the first key_transform
    converted_key_transform_list = [np.linalg.pinv(key_transform_list[0]) @ key_transform for key_transform in key_transform_list]

    key_pose_list = []
    action_pose_list = []
    # Convert list of transform matrices and actions to list of (x, y, z)
    for key_transform, action in zip(converted_key_transform_list, action_value_list):
        # Get x, y, z of key in the frame of the first key pose
        key_xyz = key_transform[0:3, -1]

        # Convert action to x, y, z in the frame of the first key pose
        # Due to line 611, 612 of envs/long_open_lock.py
        action = key_xyz - action

        key_pose_list.append(key_xyz)
        action_pose_list.append(action)

    # Store to EpisodeDemoData
    ee_init_world_pose = key_transform_list[0][0:3, -1]
    episode_demo_data = EpisodeDemoData(
        l_marker_flow=np.array(l_marker_list),
        r_marker_flow=np.array(r_marker_list),
        ee_poses=np.array(key_pose_list),
        actions=np.array(action_pose_list),
        ee_init_world_pose=ee_init_world_pose,
        actions_relative=np.array(action_value_list),
        max_action_relative=np.array(max_action) * 1e-3
    )

    # Visualize the trajectory
    # import matplotlib.pyplot as plt

    # ee_poses = episode_demo_data.ee_poses
    # actions = episode_demo_data.actions

    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(projection='3d')

    # ax.scatter(ee_poses[:, 0], ee_poses[:, 1], ee_poses[:, 2], color="r", label="ee_pose")
    # ax.scatter(actions[:, 0], actions[:, 1], actions[:, 2], color="b", label="action")
    # ax.legend()
    # for i in range(ee_poses.shape[0]):
    #     ax.text(ee_poses[i, 0], ee_poses[i, 1], ee_poses[i, 2], str(i))
    #     ax.text(actions[i, 0], actions[i, 1], actions[i, 2], str(i))
    # plt.show()

    return episode_demo_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_offsets", type=int, required=False, default=1000, help="Number of key offsets")
    args = parser.parse_args()
    num_of_offsets = args.num_of_offsets

    model = OpenLockSimpleAgent(0.7, 0.7, 0.5)
    demo_generation(model, num_of_offsets)

