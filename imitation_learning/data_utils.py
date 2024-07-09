import os
import pickle
import gzip
import numpy as np
from typing import List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EpisodeDemoData:
    l_marker_flow: np.ndarray
    r_marker_flow: np.ndarray
    actions: np.ndarray
    # ee_pose: List[np.ndarray]
    # done_flag: List[bool]


def store_data(data, file_name, datasets_dir="./data"):
    # data is a list of EpisodeDemoData objects
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, f"{file_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pkl.gzip")
    f = gzip.open(data_file, "wb")
    pickle.dump(data, f)


def read_data(file_name, datasets_dir="./data"):
    data_file = os.path.join(datasets_dir, file_name)
    f = gzip.open(data_file, "rb")
    data = pickle.load(f)
    return data


def stack_markers(markers:np.ndarray):
    # markers have shape: 2x128x2 ((original, displaced) x num_of_markers x (x, y))
    # Output shape: 128x4
    markers_reshaped = np.concatenate([markers[0], markers[1]], axis=-1)
    return markers_reshaped

