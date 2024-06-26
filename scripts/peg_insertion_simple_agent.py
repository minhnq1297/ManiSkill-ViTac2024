import numpy as np
import spatialmath as sm
# The peg insertion environment gives ground truth as obs["gt_offset"], including:
# - Current x offset of the peg w.r.t the hole
# - Current y offset of the peg w.r.t the hole
# - Current theta offset of the peg w.r.t the hole

# The peg insertion environment takes action [a_x, a_y, a_theta] to move the peg

# Simple P controller with threshold
class PegInsertionSimpleAgent():
    def __init__(self, k_x, k_y, k_theta):
        self.k_x = k_x
        self.k_y = k_y
        self.k_theta = k_theta
        self.max_x_action = 1.0
        self.max_y_action = 1.0
        self.max_theta_action = 1.0

    def set_max_action(self, max_action):
        self.max_x_action = max_action[0]
        self.max_y_action = max_action[1]
        self.max_theta_action = max_action[2]

    def predict(self, observation):
        # Output of environment in meter and radian
        peg_transform = observation["peg_transform"]
        peg_transform = sm.SE3.Rt(peg_transform[0:3, 0:3], peg_transform[0:3, -1], check=False)
        peg_offset = peg_transform.t * 1000
        peg_rpy = peg_transform.rpy() * 180 / np.pi

        # x, y: mm, theta: degree
        # This action is in the frame attached to the peg
        x_offset = peg_offset[0]
        y_offset = peg_offset[1]
        theta_offset = peg_rpy[2]
        theta_offset_rad = theta_offset * np.pi / 180.0
        delta_x = x_offset * np.cos(theta_offset_rad) + y_offset * np.sin(theta_offset_rad)
        delta_y = -x_offset * np.sin(theta_offset_rad) + y_offset * np.cos(theta_offset_rad)
        a_x = (1 if x_offset < 0.3 else self.k_x) * (0 - delta_x) / self.max_x_action
        a_y = (1 if y_offset < 0.3 else self.k_y) * (0 - delta_y) / self.max_y_action
        a_theta = (1 if theta_offset < 1.0 else self.k_theta) * (0 - theta_offset) / self.max_theta_action

        return np.array([a_x, a_y, a_theta])
