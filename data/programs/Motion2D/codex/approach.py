import numpy as np
import math


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.types = {name: observation_space.get_type(name) for name in ("crv_robot", "rectangle", "target_region")}

    def _objects(self, state, typename):
        try:
            return list(state.get_objects(self.types[typename]))
        except Exception:
            return []

    def reset(self, state, info):
        robot = self._objects(state, "crv_robot")[0]
        target = self._objects(state, "target_region")[0]
        rx = float(state.get(robot, "x"))
        radius = float(state.get(robot, "base_radius"))
        tx = float(state.get(target, "x"))
        walls = {}
        for obj in self._objects(state, "rectangle"):
            x = float(state.get(obj, "x"))
            if x <= rx + 0.02 or x >= tx - 0.02:
                continue
            walls.setdefault(round(x, 5), []).append(obj)
        self.waypoints = []
        for x, pieces in sorted(walls.items()):
            spans = sorted((float(state.get(o, "y")),
                            float(state.get(o, "y")) + float(state.get(o, "height")))
                           for o in pieces)
            # Layouts use wall segments from the lower and upper boundaries.
            if len(spans) >= 2:
                lower = min(spans, key=lambda z: z[0])
                upper = max(spans, key=lambda z: z[0])
                safe_low = max(radius, lower[1] + radius)
                safe_high = min(2.5 - radius, upper[0] - radius)
                gap_y = (safe_low + safe_high) * 0.5
                # Approach the opening before crossing: a direct diagonal aimed
                # beyond the wall can meet the wall before reaching gap height.
                self.waypoints.append((x - 0.13, gap_y))
                self.waypoints.append((x + 0.13, gap_y))
        # Rectangle poses are lower-left corners, including the target.
        self.waypoints.append((float(state.get(target, "x")) + 0.5 * float(state.get(target, "width")),
                               float(state.get(target, "y")) + 0.5 * float(state.get(target, "height"))))
        self.index = 0

    def get_action(self, state):
        robot = self._objects(state, "crv_robot")[0]
        x = float(state.get(robot, "x"))
        y = float(state.get(robot, "y"))
        while self.index < len(self.waypoints) - 1:
            wx, wy = self.waypoints[self.index]
            # Tight, boundary-clipped gaps can have sub-millimetre clearance.
            # Reach the opening center accurately before crossing horizontally.
            if abs(x - wx) < 0.001 and abs(y - wy) < 0.001:
                self.index += 1
            else:
                break
        wx, wy = self.waypoints[self.index]
        dx, dy = wx - x, wy - y
        # Keep the projecting arm downward.  The base alone determines wall
        # collision, but the arm must remain inside the arena at top-edge gaps.
        theta = float(state.get(robot, "theta"))
        angle_err = (-theta + math.pi) % (2 * math.pi) - math.pi
        return np.asarray([np.clip(dx, -0.05, 0.05), np.clip(dy, -0.05, 0.05),
                           np.clip(angle_err, -0.1963495, 0.1963495), -0.1, 0.], dtype=np.float32)
