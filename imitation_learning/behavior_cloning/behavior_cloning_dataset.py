import pickle
import gzip
import torch
import numpy as np

class BehaviorCloningDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
    ) -> None:
        f = gzip.open(data_path, "rb")
        dataset_root = pickle.load(f)
        train_l_marker_flow = dataset_root["data"]["l_marker_flow"]
        train_r_marker_flow = dataset_root["data"]["r_marker_flow"]
        train_actions = dataset_root["debug"]["actions_relative"]

        train_data = {
            "l_marker_flow": train_l_marker_flow,
            "r_marker_flow": train_r_marker_flow,
            "actions": train_actions
        }

        self.train_data = train_data
        self.len = np.shape(train_actions)[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = {
            "l_marker_flow": torch.from_numpy(self.train_data["l_marker_flow"][idx]),
            "r_marker_flow": torch.from_numpy(self.train_data["r_marker_flow"][idx]),
            "actions": torch.from_numpy(self.train_data["actions"][idx])
        }
        return sample

