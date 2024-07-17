import os
import sys
import numpy as np
import torch
import torch.nn as nn
import wandb

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from tqdm.auto import tqdm
from datetime import datetime

script_path = os.path.dirname(os.path.realpath(__file__))
repo_path = os.path.join(script_path, "..")
sys.path.append(script_path)
sys.path.insert(0, repo_path)

from imitation_learning.data_utils import ViTacDemoDataset
from solutions.networks import PointNetFeatureExtractor
from utils.common import save_checkpoint


DEBUG = 0
# Hyperparams
learning_rate = 1e-4
num_epochs = 10
batch_size = 32
checkpoint_interval = 2
# Dimensions
obs_horizon = 2
pred_horizon = 4
vision_feature_dim = 64
robot_pose_dim = 0
obs_dim = vision_feature_dim + robot_pose_dim
action_dim = 3
# Network setup
noise_pred_net = ConditionalUnet1D(input_dim=action_dim,
                                   global_cond_dim=obs_dim*obs_horizon, cond_predict_scale=True)
vision_encoder = PointNetFeatureExtractor(dim=4, out_dim=32)
# Load and freeze pretrained encoder
vision_encoder.load_state_dict(torch.load("./pretrain_weight/pretrain_peg_insertion/marker_encoder_peg_insertion.zip"))
for param in vision_encoder.parameters():
    param.requires_grad = False
vision_encoder.eval()


nets = nn.ModuleDict({
    'visual_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
})


ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75
)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=nets.parameters(),
    lr=learning_rate, weight_decay=1e-6)

dataset_path = "peg_insertion_demo-20240711-122025.pkl.gzip"
dataset = ViTacDemoDataset(
    data_path=dataset_path,
    n_obs_steps=obs_horizon,
    n_pred_steps=pred_horizon
)

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

# for this demo, we use DDPMScheduler with 100 diffusion iterations
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

device = torch.device('cuda')
nets.to(device)
ema.to(device)

if DEBUG:
    # visualize data in batch
    batch = next(iter(dataloader))
    print("batch['l_marker_flow'].shape:", batch['l_marker_flow'].shape)
    print("batch['r_marker_flow'].shape:", batch['r_marker_flow'].shape)
    print("batch['actions'].shape", batch['actions'].shape)

    marker_l = torch.flatten(batch['l_marker_flow'], start_dim=0, end_dim=1).to(device)
    marker_r = torch.flatten(batch['r_marker_flow'], start_dim=0, end_dim=1).to(device)
    marker_l_fea = vision_encoder(marker_l)
    marker_r_fea = vision_encoder(marker_r)
    marker_fea = torch.cat((marker_l_fea, marker_r_fea), dim=1)

wandb.init(
    project="dl_lab_mani_vitac",
    name=f"train_peg_insertion_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    config={
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "obs_horizon": obs_horizon,
        "pred_horizon": pred_horizon
    }
)

if not DEBUG:
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    left_markers = nbatch['l_marker_flow'][:, :obs_horizon]
                    right_markers = nbatch['r_marker_flow'][:, :obs_horizon]
                    actions = nbatch['actions'].to(device)
                    B = actions.shape[0]
                    marker_l = torch.flatten(left_markers, start_dim=0, end_dim=1).to(device)
                    marker_r = torch.flatten(right_markers, start_dim=0, end_dim=1).to(device)
                    marker_l_fea = nets['visual_encoder'](marker_l)
                    marker_r_fea = nets['visual_encoder'](marker_r)
                    marker_fea = torch.cat((marker_l_fea, marker_r_fea), dim=1)

                    # encoder vision features
                    image_features = marker_fea.reshape((B, obs_horizon, marker_fea.shape[1]))
                    # (B,obs_horizon,D)

                    # concatenate vision feature and low-dim obs
                    # obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = image_features.flatten(start_dim=1).to(device)
                    # (B, obs_horizon * obs_dim)

                    # sample noise to add to actions
                    noise = torch.randn(actions.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        actions, noise, timesteps)

                    # predict the noise residual
                    noise_pred = nets['noise_pred_net'](
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            mean_epoch_loss = np.mean(epoch_loss)
            tglobal.set_postfix(loss=mean_epoch_loss)
            wandb.log(
                {
                    "loss": mean_epoch_loss
                }
            )

            # Save model checkpoint every checkpoint interval
            if(epoch_idx + 1) % checkpoint_interval == 0:
                checkpoint = {
                    'epoch': epoch_idx + 1,
                    'noise_pred_net_statedict': nets['noise_pred_net'].state_dict(),
                    'visual_encoder_statedict': nets['visual_encoder'].state_dict(),
                    # 'optimizer_statedict': optimizer.state_dict(),
                }
                save_checkpoint(checkpoint)


    print('Finished Training')