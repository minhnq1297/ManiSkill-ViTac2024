import copy
import os
import sys
import numpy as np
import ruamel.yaml as yaml
from loguru import logger

script_path = os.path.dirname(os.path.realpath(__file__))
repo_path = os.path.join(script_path, "..")
sys.path.append(script_path)
sys.path.insert(0, repo_path)

from scripts.arguments import parse_params, handle_policy_args
from envs.long_open_lock import LongOpenLockRandPointFlowEnv
from path import Path
from stable_baselines3.common.utils import set_random_seed
from utils.common import get_time, get_average_params
from loguru import logger

from imitation_learning.open_lock_simple_agent import OpenLockSimpleAgent
from imitation_learning.data_utils import *

import matplotlib.pyplot as plt

EVAL_CFG_FILE = os.path.join(repo_path, "configs/parameters/long_open_lock_demo_gen.yaml")
KEY_NUM = 2
REPEAT_NUM = 1

def demo_generation(model):
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
        model.set_max_action(cfg["env"]["max_action"])

    specified_env_args = copy.deepcopy(cfg["env"])

    specified_env_args.update(
        {
            "params": average_params,
            "params_upper_bound": average_params,
        }
    )

    # create evaluation environment
    env = LongOpenLockRandPointFlowEnv(**specified_env_args)
    set_random_seed(0)


    max_key_x_offset = cfg["env"]["key_x_max_offset"]
    max_key_y_offset = cfg["env"]["key_y_max_offset"]
    max_key_z_offset = cfg["env"]["key_z_max_offset"]
    num_of_offsets = 3
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
                action_list = []

                marker_pos = o["marker_flow"]
                l_marker_pos, r_marker_pos = marker_pos[0], marker_pos[1]
                l_marker_list.append(stack_markers(l_marker_pos))
                r_marker_list.append(stack_markers(r_marker_pos))

                d, ep_ret, ep_len = False, 0, 0
                while not d:
                    ep_len += 1
                    action = model.predict(o)
                    logger.info(f"Step {ep_len} Action: {action}")
                    o, r, terminated, truncated, info = env.step(action)
                    d = terminated or truncated
                    ep_ret += r

                    marker_pos_sub_steps = o["marker_flow_sub_steps"]
                    for marker_pos_sub_step in marker_pos_sub_steps:
                        action_list.append(action)
                        l_marker_pos_sub_step, r_marker_pos_sub_step = marker_pos_sub_step[0], marker_pos_sub_step[1]
                        l_marker_list.append(stack_markers(l_marker_pos_sub_step))
                        r_marker_list.append(stack_markers(r_marker_pos_sub_step))

                    # marker_pos = o["marker_flow"]
                    # l_marker_pos, r_marker_pos = marker_pos[0], marker_pos[1]
                    # l_marker_list.append(stack_markers(l_marker_pos))
                    # r_marker_list.append(stack_markers(r_marker_pos))
                    # action_list.append(action)

                if info["is_success"]:
                    collect_result.append([True, ep_len])
                    logger.opt(colors=True).info(f"<green>RESULT: SUCCESS</green>")
                    action_list.append(np.zeros_like(action))
                    episode_demo_data = EpisodeDemoData(
                        np.array(l_marker_list),
                        np.array(r_marker_list),
                        np.array(action_list)
                    )
                    episode_demo_data_list.append(episode_demo_data)
                else:
                    collect_result.append([False, ep_len])
                    logger.opt(colors=True).info(f"<d>RESULT: FAIL</d>")

    store_data(episode_demo_data_list, "open_lock_demo")

    env.close()
    success_rate = np.sum(np.array([int(v[0]) for v in collect_result])) / (episode_num * KEY_NUM * REPEAT_NUM)
    if success_rate > 0:
        avg_steps = np.mean(np.array([int(v[1]) if v[0] else 0 for v in collect_result])) / success_rate
        logger.info(f"#SUCCESS_RATE: {success_rate*100.0:.2f}%")
        logger.info(f"#AVG_STEP: {avg_steps:.2f}")
    else:
        logger.info(f"#SUCCESS_RATE: 0")
        logger.info(f"#AVG_STEP: NA")



if __name__ == "__main__":
    model = OpenLockSimpleAgent(0.7, 0.7, 0.5)
    demo_generation(model)

