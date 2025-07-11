# trajectory.py
import numpy as np

class Trajectory:
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.current_index = 0

    def get_target(self):
        if self.current_index < len(self.waypoints):
            return self.waypoints[self.current_index]
        return None

    def advance_if_reached(self, x, y, threshold=0.1):
        target = self.get_target()
        if target is None:
            return
        distance = np.linalg.norm(np.array([x,y]) - np.array(target))
        if distance < threshold:
            self.current_index += 1

    def is_finished(self):
        return self.current_index >= len(self.waypoints)
