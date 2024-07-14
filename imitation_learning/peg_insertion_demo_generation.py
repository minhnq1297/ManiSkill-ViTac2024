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

from scripts.arguments import parse_params
from envs.peg_insertion import ContinuousInsertionSimGymRandomizedPointFLowEnv
from path import Path
from stable_baselines3.common.utils import set_random_seed
from utils.common import get_time, get_average_params
from loguru import logger

from imitation_learning.peg_insertion_simple_agent import PegInsertionSimpleAgent
from imitation_learning.data_utils import *

DATA_GEN_CFG_FILE = os.path.join(repo_path, "configs/parameters/peg_insertion_demo_gen.yaml")
PEG_NUM = 2
REPEAT_NUM = 1

def demo_generation(model):
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
        model.set_max_action(cfg["env"]["max_action"])

    specified_env_args = copy.deepcopy(cfg["env"])

    specified_env_args.update(
        {
            "params": average_params,
            "params_upper_bound": average_params,
        }
    )

    # create evaluation environment
    env = ContinuousInsertionSimGymRandomizedPointFLowEnv(**specified_env_args)
    set_random_seed(0)

    # Generate offset_list randomly: x,y in mm, theta in degree
    max_x_offset = cfg["env"]["peg_x_max_offset"]
    max_y_offset = cfg["env"]["peg_y_max_offset"]
    max_theta_offset = cfg["env"]["peg_theta_max_offset"]
    num_of_offsets = 2
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
                action_list = []

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

                    obs_sub_steps = o["obs_sub_steps"]
                    for obs_sub_step in obs_sub_steps:
                        action_sub_step = model.predict(obs_sub_step)
                        action_list.append(action_sub_step)
                        marker_pos_sub_step = obs_sub_step["marker_flow"]
                        l_marker_pos_sub_step, r_marker_pos_sub_step = marker_pos_sub_step[0], marker_pos_sub_step[1]
                        l_marker_list.append(stack_markers(l_marker_pos_sub_step))
                        r_marker_list.append(stack_markers(r_marker_pos_sub_step))

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
                    episode_demo_data = EpisodeDemoData(
                        np.array(l_marker_list),
                        np.array(r_marker_list),
                        np.array(action_list)
                    )
                    episode_demo_data_list.append(episode_demo_data)
                else:
                    collect_result.append([False, ep_len])
                    logger.opt(colors=True).info(f"<d>RESULT: FAIL</d>")

    store_data(episode_demo_data_list, "peg_insertion_demo")

    env.close()
    success_rate = np.sum(np.array([int(v[0]) for v in collect_result])) / (episode_num * PEG_NUM * REPEAT_NUM)
    if success_rate > 0:
        avg_steps = np.mean(np.array([int(v[1]) if v[0] else 0 for v in collect_result])) / success_rate
        logger.info(f"#SUCCESS_RATE: {success_rate*100.0:.2f}%")
        logger.info(f"#AVG_STEP: {avg_steps:.2f}")
    else:
        logger.info(f"#SUCCESS_RATE: 0")



if __name__ == "__main__":
    model = PegInsertionSimpleAgent(1.5, 1.5, 1.0)
    demo_generation(model)
