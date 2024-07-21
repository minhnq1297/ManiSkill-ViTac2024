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

from scripts.arguments import parse_params
from envs.peg_insertion import ContinuousInsertionSimGymRandomizedPointFLowEnv
from path import Path
from stable_baselines3.common.utils import set_random_seed
from utils.common import get_time, get_average_params
from loguru import logger

from imitation_learning.data_generation.peg_insertion_simple_agent import PegInsertionSimpleAgent
from imitation_learning.utils import *

DATA_GEN_CFG_FILE = os.path.join(repo_path, "configs/parameters/peg_insertion_demo_gen.yaml")
PEG_NUM = 3
REPEAT_NUM = 1

def demo_generation(model, num_of_offsets):
    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}", level="INFO")

    with open(DATA_GEN_CFG_FILE, "r") as f:
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

    env = ContinuousInsertionSimGymRandomizedPointFLowEnv(**specified_env_args)
    set_random_seed(0)

    # Generate offset_list randomly: x,y in mm, theta in degree
    max_x_offset = cfg["env"]["peg_x_max_offset"]
    max_y_offset = cfg["env"]["peg_y_max_offset"]
    max_theta_offset = cfg["env"]["peg_theta_max_offset"]
    max_offset = np.array([max_x_offset, max_y_offset, max_theta_offset])

    offset_list = -max_offset + 2 * max_offset * np.random.rand(num_of_offsets, 3)
    offset_list = offset_list.tolist()

    episode_num = len(offset_list)
    collect_result = []

    episode_demo_data_list = []
    for i in range(PEG_NUM):
        for r in range(REPEAT_NUM):
            for k in range(episode_num):
                logger.opt(colors=True).info(f"<blue>#### Run No. {len(collect_result) + 1} ####</blue>")
                o, _ = env.reset(offset_list[k], peg_idx=i)

                l_marker_list = []
                r_marker_list = []
                peg_transform_list = []
                action_list = []

                peg_transform = o["peg_transform"]
                peg_transform_list.append(peg_transform)
                marker_pos = o["marker_flow"]
                l_marker_pos, r_marker_pos = marker_pos[0], marker_pos[1]
                l_marker_list.append(stack_markers(l_marker_pos))
                r_marker_list.append(stack_markers(r_marker_pos))

                initial_offset_of_current_episode = o["gt_offset"]
                logger.info(f"Initial offset: {initial_offset_of_current_episode}")
                d, ep_ret, ep_len = False, 0, 0
                while not d:
                    ep_len += 1
                    action = model.predict(o)
                    action_list.append(action)
                    logger.info(f"Step {ep_len} Action: {action}")
                    o, r, terminated, truncated, info = env.step(action)
                    d = terminated or truncated

                    # obs_sub_steps = o["obs_sub_steps"]
                    # for obs_sub_step in obs_sub_steps:
                    #     action_sub_step = model.predict(obs_sub_step)
                    #     action_list.append(action_sub_step)
                    #     peg_transform_sub_step = obs_sub_step["peg_transform"]
                    #     peg_transform_list.append(peg_transform_sub_step)
                    #     marker_pos_sub_step = obs_sub_step["marker_flow"]
                    #     l_marker_pos_sub_step, r_marker_pos_sub_step = marker_pos_sub_step[0], marker_pos_sub_step[1]
                    #     l_marker_list.append(stack_markers(l_marker_pos_sub_step))
                    #     r_marker_list.append(stack_markers(r_marker_pos_sub_step))

                    peg_transform = o["peg_transform"]
                    peg_transform_list.append(peg_transform)
                    marker_pos = o["marker_flow"]
                    l_marker_pos, r_marker_pos = marker_pos[0], marker_pos[1]
                    l_marker_list.append(stack_markers(l_marker_pos))
                    r_marker_list.append(stack_markers(r_marker_pos))

                    if 'gt_offset' in o.keys():
                        logger.info(f"Offset: {o['gt_offset']}")
                    ep_ret += r

                if info["is_success"]:
                    collect_result.append([True, ep_len])
                    logger.opt(colors=True).info(f"<green>RESULT: SUCCESS</green>")
                    action_list.append(np.zeros_like(action))
                    episode_demo_data = preprocessing_episode_data(l_marker_list, r_marker_list,
                                                                   peg_transform_list, action_list, max_action)
                    episode_demo_data_list.append(episode_demo_data)
                else:
                    collect_result.append([False, ep_len])
                    logger.opt(colors=True).info(f"<d>RESULT: FAIL</d>")

    normalize_and_store_data(episode_demo_data_list, "peg_insertion_demo")

    env.close()
    success_rate = np.sum(np.array([int(v[0]) for v in collect_result])) / (episode_num * PEG_NUM * REPEAT_NUM)
    if success_rate > 0:
        avg_steps = np.mean(np.array([int(v[1]) if v[0] else 0 for v in collect_result])) / success_rate
        logger.info(f"#SUCCESS_RATE: {success_rate*100.0:.2f}%")
        logger.info(f"#AVG_STEP: {avg_steps:.2f}")
    else:
        logger.info(f"#SUCCESS_RATE: 0")


def preprocessing_episode_data(l_marker_list, r_marker_list, peg_transform_list, action_list, max_action):
    # Convert action back to normal value in mm and degree
    action_value_list = [action_percentage_to_value(action, max_action) for action in action_list]

    # We always consider the first transform of the peg as the origin for others
    # Convert transform w.r.t the first peg_transform
    converted_peg_transform_list = [np.linalg.pinv(peg_transform_list[0]) @ peg_transform for peg_transform in peg_transform_list]

    peg_pose_list = []
    action_pose_list = []
    # Convert list of transform matrices and actions to list of (x, y, theta)
    for peg_transform, action in zip(converted_peg_transform_list, action_value_list):
        # Get x, y, theta of peg in the frame of the first peg pose
        peg_xyz, peg_rpy = transformation_matrix_to_xyz_rpy(peg_transform)
        # Convert peg positions to mm
        peg_x = peg_xyz[0] * 1000.0
        peg_y = peg_xyz[1] * 1000.0
        peg_theta = peg_rpy[-1]

        # Convert action to x, y, theta in the frame of the first peg pose
        action_x = action[0] * np.cos(peg_theta) - action[1] * np.sin(peg_theta)
        action_y = action[0] * np.sin(peg_theta) + action[1] * np.cos(peg_theta)
        action_theta = action[2]

        action_x += peg_x
        action_y += peg_y
        # Convert peg orientation to degree
        peg_theta = peg_theta * 180.0 / np.pi
        action_theta += peg_theta

        peg_pose_list.append(np.array([peg_x, peg_y, peg_theta]))
        action_pose_list.append(np.array([action_x, action_y, action_theta]))

    # Store to EpisodeDemoData
    ee_init_world_xyz, ee_init_world_rpy = transformation_matrix_to_xyz_rpy(peg_transform_list[0])
    ee_init_world_pose = np.array([
        ee_init_world_xyz[0] * 1000.0,
        ee_init_world_xyz[1] * 1000.0,
        ee_init_world_rpy[2] * 180.0 / np.pi
    ])
    episode_demo_data = EpisodeDemoData(
        l_marker_flow=np.array(l_marker_list),
        r_marker_flow=np.array(r_marker_list),
        ee_poses=np.array(peg_pose_list),
        actions=np.array(action_pose_list),
        ee_init_world_pose=ee_init_world_pose,
        actions_relative=np.array(action_value_list),
        max_action_relative=np.array(max_action)
    )

    return episode_demo_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_offsets", type=int, required=False, default=1000, help="Number of peg offsets")
    args = parser.parse_args()
    num_of_offsets = args.num_of_offsets

    model = PegInsertionSimpleAgent(1.5, 1.5, 1.0)
    demo_generation(model, num_of_offsets)
