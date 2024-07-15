import os
import sys
import copy
import collections
import torch
import numpy as np
import torch.nn as nn
import ruamel.yaml as yaml

from path import Path
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from tqdm.auto import tqdm
from stable_baselines3.common.utils import set_random_seed

script_path = os.path.dirname(os.path.realpath(__file__))
repo_path = os.path.join(script_path, "..")
sys.path.append(script_path)
sys.path.insert(0, repo_path)

from envs.peg_insertion import ContinuousInsertionSimGymRandomizedPointFLowEnv
from loguru import logger
from imitation_learning.data_utils import ViTacDemoDataset
from solutions.networks import PointNetFeatureExtractor
from scripts.arguments import parse_params
from utils.common import get_time, get_average_params

EVAL_CFG_FILE = os.path.join(repo_path, "configs/evaluation/peg_insertion_evaluation.yaml")
PEG_NUM = 3
REPEAT_NUM = 2
DEVICE = torch.device('cuda')


def observation_to_features(feature_extractor_net, original_obs) -> torch.Tensor:
    if original_obs.ndim == 4:
        original_obs = torch.unsqueeze(original_obs, 0)
    batch_num = original_obs.shape[0]
    # (batch_num, 2 (left_and_right), 2 (no-contact and contact), 128 (marker_num), 2 (u, v))
    feature_extractor_input = torch.cat([original_obs[:, :, 0, ...], original_obs[:, :, 1, ...]], dim=-1)
    feature_extractor_input = torch.cat([feature_extractor_input[:, 0, ...], feature_extractor_input[:, 1, ...]], dim=0)
    # (batch_num * 2, 128, 4)
    # l_marker_pos = feature_extractor_input[:, 0, ...]
    # r_marker_pos = feature_extractor_input[:, 1, ...]
    # shape: (batch, num_points, 4)

    marker_flow_fea = feature_extractor_net(feature_extractor_input)
    # l_marker_flow_fea = self.feature_extractor_net(l_marker_pos)
    # r_marker_flow_fea = self.feature_extractor_net(r_marker_pos)  # (batch_num, pointnet_feature_dim)
    marker_flow_fea = torch.cat([marker_flow_fea[:batch_num], marker_flow_fea[batch_num:]], dim=-1)

    return marker_flow_fea

def evaluate_policy(model, noise_scheduler, action_dim, pred_horizon, obs_horizon, action_horizon, render_rgb):
    exp_start_time = get_time()
    exp_name = f"peg_insertion_{exp_start_time}"
    log_dir = Path(os.path.join(repo_path, f"eval_log/{exp_name}"))
    log_dir.makedirs_p()

    logger.remove()
    logger.add(log_dir / f"{exp_name}.log")
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

    specified_env_args = copy.deepcopy(cfg["env"])

    specified_env_args.update(
        {
            "params": average_params,
            "params_upper_bound": average_params,
        }
    )
    specified_env_args["render_rgb"] = render_rgb

    # create evaluation environment
    env = ContinuousInsertionSimGymRandomizedPointFLowEnv(**specified_env_args)
    set_random_seed(0)

    offset_list = [[-4.0, -4.0, -8.0], [-4.0, -2.0, 2.0], [-4.0, 1.0, -6.0], [-4.0, 3.0, 6.0], [-3.0, -3.0, -2.0],
                   [-3.0, -1.0, 8.0], [-3.0, 2.0, 2.0], [-2.0, -4.0, -6.0], [-2.0, -2.0, 4.0], [-2.0, 1.0, -2.0],
                   [-2.0, 3.0, 8.0], [-1.0, -3.0, 0.0], [-1.0, 0.0, 6.0], [-1.0, 3.0, 4.0], [0.0, -3.0, -4.0],
                   [0.0, 0.0, 6.0], [0.0, 3.0, 4.0], [1.0, -3.0, -4.0], [1.0, 0.0, -4.0], [1.0, 3.0, 0.0],
                   [2.0, -3.0, -8.0], [2.0, -1.0, 4.0], [2.0, 2.0, -4.0], [2.0, 4.0, 6.0], [3.0, -2.0, 0.0],
                   [3.0, 1.0, -8.0], [3.0, 3.0, 2.0], [4.0, -3.0, -4.0], [4.0, -1.0, 6.0], [4.0, 2.0, -2.0]]
    test_num = len(offset_list)
    test_result = []

    for i in range(PEG_NUM):
        for r in range(REPEAT_NUM):
            for k in range(test_num):
                logger.opt(colors=True).info(f"<blue>#### Test No. {len(test_result) + 1} ####</blue>")
                o, _ = env.reset(offset_list[k], peg_idx=i)
                initial_offset_of_current_episode = o["gt_offset"]
                logger.info(f"Initial offset: {initial_offset_of_current_episode}")
                d, ep_ret, ep_len = False, 0, 0
                
                obs_deque = collections.deque([o] * obs_horizon, maxlen=obs_horizon)
                while not d:
                    # Take deterministic actions at test time (noise_scale=0)
                    ep_len += 1

                    with torch.no_grad():
                        markers = torch.from_numpy(np.stack([obs["marker_flow"] for obs in obs_deque]))
                        markers = markers.to(device=DEVICE)
                        obs_features = observation_to_features(model["visual_encoder"], markers)
                        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
                        noisy_action = torch.randn(
                            (1, pred_horizon, action_dim), device=DEVICE)
                        naction = noisy_action

                        for k in noise_scheduler.timesteps:
                            # predict noise
                            noise_pred = model["noise_pred_net"](
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

                    #TODO: review this again
                    naction = naction.cpu().detach().numpy()
                    naction = naction[0]
                    start = obs_horizon - 1
                    end = start + action_horizon
                    action = naction[start:end,:]
                    # (action_horizon, action_dim)

                    # TODO: edit this to denoise process
                    logger.info(f"Step {ep_len} Action: {action}")

                    #assume now only take 1 action
                    o, r, terminated, truncated, info = env.step(action)
                    obs_deque.append(o)
                    d = terminated or truncated
                    if 'gt_offset' in o.keys():
                        logger.info(f"Offset: {o['gt_offset']}")
                    ep_ret += r
                if info["is_success"]:
                    test_result.append([True, ep_len])
                    logger.opt(colors=True).info(f"<green>RESULT: SUCCESS</green>")
                else:
                    test_result.append([False, ep_len])
                    logger.opt(colors=True).info(f"<d>RESULT: FAIL</d>")

    env.close()
    success_rate = np.sum(np.array([int(v[0]) for v in test_result])) / (test_num * PEG_NUM * REPEAT_NUM)
    if success_rate > 0:
        avg_steps = np.mean(np.array([int(v[1]) if v[0] else 0 for v in test_result])) / success_rate
        logger.info(f"#SUCCESS_RATE: {success_rate*100.0:.2f}%")
        logger.info(f"#AVG_STEP: {avg_steps:.2f}")
    else:
        logger.info(f"#SUCCESS_RATE: 0")
        logger.info(f"#AVG_STEP: NA")


if __name__ == "__main__":
    # Dimensions
    obs_horizon = 2
    action_horizon = 1
    pred_horizon = 4
    vision_feature_dim = 64
    robot_pose_dim = 0
    obs_dim = vision_feature_dim + robot_pose_dim
    action_dim = 3
    # Network setup
    noise_pred_net = ConditionalUnet1D(input_dim=action_dim,
                                    global_cond_dim=obs_dim*obs_horizon, cond_predict_scale=True)
    vision_encoder = PointNetFeatureExtractor(dim=4, out_dim=32)
    # Load checkpoint
    check_point = torch.load("./trained_model/checkpoint_peg_model.pth.tar")
    # Load and freeze trained Diffusion network
    noise_pred_net_state_dict = check_point["noise_pred_net_statedict"]
    noise_pred_net.load_state_dict(noise_pred_net_state_dict)
    noise_pred_net.eval()

    # Load and freeze pretrained encoder
    vision_encoder_state_dict = torch.load("./pretrain_weight/pretrain_peg_insertion/marker_encoder_peg_insertion.zip")
    vision_encoder.load_state_dict(vision_encoder_state_dict)
    vision_encoder.eval()

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

    evaluate_policy(nets, noise_scheduler, action_dim, pred_horizon, obs_horizon, action_horizon, False)

