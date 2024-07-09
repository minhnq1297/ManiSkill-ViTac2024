import numpy as np
# The open lock environment gives ground truth as obs["key_1"], obs["key_2"], obs["lock_1"], obs["lock_2"] including:
# - Current x, y, z of 4 corner-points of key protrusions
# - Current x, y, z of 4 corner-points of lock pins

# The open lock environment takes action [a_x, a_y, a_z] to move the key

# Simple P controller with threshold
class OpenLockSimpleAgent():
    def __init__(self, k_x, k_y, k_z):
        self.k_x = k_x
        self.k_y = k_y
        self.k_z = k_z

    def set_max_action(self, max_action):
        self.max_x_action = max_action[0]
        self.max_y_action = max_action[1]
        self.max_z_action = max_action[2]

    def predict(self, observation):
        # Output of environment in m, need to convert to mm
        # key1_pts, key2_pts: key protrusions
        key1_pts = observation["key1_pts"]
        key2_pts = observation["key2_pts"]
        # lock1_pts, lock2_pts: lock pins
        lock1_pts = observation["lock1_pts"]
        lock2_pts = observation["lock2_pts"]

        # Compute differences in x, y, z of the key protrusions and pins
        key_1_x_offset = key1_pts.mean(axis=0)[0] - lock1_pts.mean(axis=0)[0]
        key_2_x_offset = key2_pts.mean(axis=0)[0] - lock2_pts.mean(axis=0)[0]
        key_x_offset = 1000 * (key_1_x_offset + key_2_x_offset) / 2

        key_1_y_offset = key1_pts.mean(axis=0)[1] - lock1_pts.mean(axis=0)[1]
        key_2_y_offset = key2_pts.mean(axis=0)[1] - lock2_pts.mean(axis=0)[1]
        key_y_offset = 1000 * (key_1_y_offset + key_2_y_offset) / 2

        key_1_z_offset = key1_pts.mean(axis=0)[2] - lock1_pts.mean(axis=0)[2]
        key_2_z_offset = key2_pts.mean(axis=0)[2] - lock2_pts.mean(axis=0)[2]
        key_z_offset = 1000 * (key_1_z_offset + key_2_z_offset) / 2

        a_x = self.k_x * key_x_offset
        a_x = np.clip(a_x, -self.max_x_action, self.max_x_action) / self.max_x_action
        a_y = self.k_y * key_y_offset
        a_y = np.clip(a_y, -self.max_y_action, self.max_y_action) / self.max_y_action
        # If the x-y offset stil large, do not move the key along z
        key_x_y_offset = np.sqrt(key_x_offset ** 2 + key_y_offset ** 2)
        if key_x_y_offset < 0.7:
            a_z = self.k_z * (key_z_offset - 0.010 * 1000)
            a_z = np.clip(a_z, -self.max_z_action, self.max_z_action) / self.max_z_action
        else:
            a_z = 0

        return np.array([a_x, a_y, a_z])

