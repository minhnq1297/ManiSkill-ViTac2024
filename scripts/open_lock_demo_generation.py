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

from scripts.open_lock_simple_agent import OpenLockSimpleAgent

import matplotlib.pyplot as plt

EVAL_CFG_FILE = os.path.join(repo_path, "configs/evaluation/open_lock_evaluation.yaml")
KEY_NUM = 1
REPEAT_NUM = 1

def demo_generation(model):
    exp_start_time = get_time()

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

    offset_list = [[i * 1.0 / 2, 0, 0] for i in range(20)]
    test_num = len(offset_list)
    test_result = []

    for i in range(KEY_NUM):
        for r in range(REPEAT_NUM):
            for k in range(test_num):
                logger.opt(colors=True).info(f"<blue>#### Test No. {len(test_result) + 1} ####</blue>")
                o, _ = env.reset(offset_list[k], key_idx=i)
                d, ep_ret, ep_len = False, 0, 0
                while not d:
                    # Take deterministic actions at test time (noise_scale=0)
                    ep_len += 1
                    action = model.predict(o)
                    logger.info(f"Step {ep_len} Action: {action}")
                    o, r, terminated, truncated, info = env.step(action)
                    # Store sensor output and action as npz. Do not need the original marker positions
                    # lr_marker_flow = o["marker_flow"]
                    # l_marker_flow, r_marker_flow = lr_marker_flow[0], lr_marker_flow[1]
                    # plt.figure(1, (20, 9))
                    # ax = plt.subplot(1, 2, 1)
                    # ax.scatter(l_marker_flow[0, :, 0], l_marker_flow[0, :, 1], c="blue")
                    # ax.scatter(l_marker_flow[1, :, 0], l_marker_flow[1, :, 1], c="red")
                    # plt.xlim(15, 315)
                    # plt.ylim(15, 235)
                    # ax.invert_yaxis()
                    # ax = plt.subplot(1, 2, 2)
                    # ax.scatter(r_marker_flow[0, :, 0], r_marker_flow[0, :, 1], c="blue")
                    # ax.scatter(r_marker_flow[1, :, 0], r_marker_flow[1, :, 1], c="red")
                    # plt.xlim(15, 315)
                    # plt.ylim(15, 235)
                    # ax.invert_yaxis()

                    d = terminated or truncated
                    ep_ret += r
                if info["is_success"]:
                    test_result.append([True, ep_len])
                    logger.opt(colors=True).info(f"<green>RESULT: SUCCESS</green>")
                else:
                    test_result.append([False, ep_len])
                    logger.opt(colors=True).info(f"<d>RESULT: FAIL</d>")

    env.close()
    success_rate = np.sum(np.array([int(v[0]) for v in test_result])) / (test_num * KEY_NUM * REPEAT_NUM)
    if success_rate > 0:
        avg_steps = np.mean(np.array([int(v[1]) if v[0] else 0 for v in test_result])) / success_rate
        logger.info(f"#SUCCESS_RATE: {success_rate*100.0:.2f}%")
        logger.info(f"#AVG_STEP: {avg_steps:.2f}")
    else:
        logger.info(f"#SUCCESS_RATE: 0")
        logger.info(f"#AVG_STEP: NA")



if __name__ == "__main__":
    model = OpenLockSimpleAgent(0.7, 0.7, 0.5)
    demo_generation(model)

