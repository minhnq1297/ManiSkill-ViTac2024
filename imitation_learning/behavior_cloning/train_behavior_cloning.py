import os
import sys
import wandb
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime

script_path = os.path.dirname(os.path.realpath(__file__))
repo_path = os.path.join(script_path, "../..")
sys.path.append(script_path)
sys.path.insert(0, repo_path)

from imitation_learning.behavior_cloning.behavior_cloning_agent import BCAgent
from imitation_learning.behavior_cloning.behavior_cloning_dataset import BehaviorCloningDataset
from solutions.networks import PointNetFeatureExtractor
from utils.common import save_checkpoint

DEVICE = torch.device('cuda')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True, help="Task names: peg_insertion or open_lock")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--use_pretrained_encoder", type=bool, required=False, default=True, help="Using pretrained encoder")
    parser.add_argument("--learning_rate", type=float, required=False, default=1e-4, help="Learning rate")
    parser.add_argument("--n_epoch", type=int, required=False, default=1000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, required=False, default=32, help="Batch size")
    parser.add_argument("--use_wandb", type=bool, required=False, default=False, help="Logging to wandb")

    args = parser.parse_args()
    task_name = args.task_name
    assert(task_name == "peg_insertion" or task_name == "open_lock")
    dataset_path = args.train_data_path
    use_pretrained_encoder = args.use_pretrained_encoder
    use_wandb = args.use_wandb

    learning_rate = args.learning_rate
    num_epochs = args.n_epoch
    batch_size = args.batch_size

    # Feature dim from pretrained encoder
    feature_dim = 64
    hidden_dim = 32
    actions_dim = 3

    # Behavior Cloning Agent setup
    bc_network = BCAgent(feature_dim=feature_dim, hidden_dim=hidden_dim, actions_dim=actions_dim).to(device=DEVICE)
    # Create optimizer
    optimizer = optim.Adam(bc_network.parameters(), lr=learning_rate)

    # Pretrained encoder setup
    vision_encoder = PointNetFeatureExtractor(dim=4, out_dim=32)
    if use_pretrained_encoder:
        if task_name == "peg_insertion":
            vision_encoder.load_state_dict(torch.load("./pretrain_weight/pretrain_peg_insertion/marker_encoder_peg_insertion.zip"))
        elif task_name == "open_lock":
            vision_encoder.load_state_dict(torch.load("./pretrain_weight/pretrain_openlock/marker_encoder_openlock.zip"))
        for param in vision_encoder.parameters():
            param.requires_grad = False
        vision_encoder.eval()
    vision_encoder.to(device=DEVICE)

    # Preparing data
    checkpoint_interval = 2

    dataset = BehaviorCloningDataset(
        data_path=dataset_path
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

    if use_wandb:
        wandb.init(
            project="dl_lab_mani_vitac",
            name=f"train_behavior_cloning_{task_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            }
        )

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    l_marker_flow = nbatch["l_marker_flow"].to(DEVICE)
                    r_marker_flow = nbatch["r_marker_flow"].to(DEVICE)
                    actions = nbatch["actions"].to(DEVICE)

                    # Pass observation to encoder
                    l_encoding = vision_encoder(l_marker_flow)
                    r_encoding = vision_encoder(r_marker_flow)
                    encoding = torch.cat([l_encoding, r_encoding], dim=1)

                    # Pass encoding to agent
                    actions_pred = bc_network(encoding)
                    batch_loss = nn.functional.mse_loss(actions_pred, actions)

                    # Run optimizer
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                    batch_loss_cpu = batch_loss.item()
                    epoch_loss.append(batch_loss_cpu)
                    tepoch.set_postfix(loss=batch_loss_cpu)
                mean_epoch_loss = np.mean(epoch_loss)
                tglobal.set_postfix(loss=mean_epoch_loss)
                if use_wandb:
                    wandb.log(
                        {
                            "loss": mean_epoch_loss
                        }
                    )

                # Save model checkpoint every checkpoint interval
                if(epoch_idx + 1) % checkpoint_interval == 0:
                    checkpoint = {
                        'epoch': epoch_idx + 1,
                        'noise_pred_net_statedict': bc_network.state_dict(),
                        'visual_encoder_statedict': vision_encoder.state_dict(),
                        # 'optimizer_statedict': optimizer.state_dict(),
                    }
                    save_checkpoint(checkpoint, filename=f"bc_checkpoint_{task_name}_model.pth.tar")


        print('Finished Training')
