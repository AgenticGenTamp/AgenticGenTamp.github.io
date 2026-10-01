import math

import numpy as np


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class GeneratedApproach:
    """Closed-loop contact planner for the asymmetric T-shaped rigid body."""

    def __init__(self, action_space, observation_space, primitives):
        # Keep commands just inside the server's native float32 Box limits.
        self.low = np.asarray(action_space.low, dtype=float) * 0.998
        self.high = np.asarray(action_space.high, dtype=float) * 0.998
        self.phase = "decide"
        self.push_steps = 0
        self.side = 1.0
        self.normal = 1.0
        self.retreat_vec = np.array([1.0, 0.0])
        self.push_origin = np.zeros(2)
        self.push_goal_dist = 0.0
        self.rotate_stroke = 7
        self.stage_steps = 0
        self.alternate_rotation = True
        self.prefer_nearest = False
        self.wide_clearance = False
        self.allow_fine_nearest = False
        self.com_fraction = 0.30
        self.rotation_bursts = 0
        self.escape_push = False
        self.allow_wall_escape = True
        self.step = 0

    def reset(self, state, info):
        self.phase = "decide"
        self.push_steps = 0
        self.side = 1.0
        self.normal = 1.0
        self.retreat_vec = np.array([1.0, 0.0])
        self.push_origin = np.zeros(2)
        self.push_goal_dist = 0.0
        self.rotate_stroke = 7
        self.stage_steps = 0
        self.alternate_rotation = True
        initial_error = abs(_wrap(float(state[31]) - float(state[2])))
        # Empirically, medium turns finish much faster when repeatedly using
        # the nearby face; very small or large turns benefit from balanced
        # alternating contacts that limit pose drift.
        self.prefer_nearest = 0.65 < initial_error < 1.40
        self.wide_clearance = initial_error > 1.90
        self.allow_fine_nearest = 0.35 < initial_error <= 1.90
        self.com_fraction = 0.40 if 1.70 < initial_error < 1.90 else 0.30
        self.rotation_bursts = 0
        self.escape_push = False
        self.allow_wall_escape = not (1.40 < initial_error < 2.10)
        self.step = 0

    @staticmethod
    def _rot(theta):
        c, sn = math.cos(theta), math.sin(theta)
        return np.array([[c, -sn], [sn, c]])

    def _move(self, robot, target):
        delta = target - robot
        # Keep the exact intended path direction: component clipping bends
        # diagonal contact lines and produces large unintended torque.
        scale = min(1.0, float(self.high[0]) /
                    max(1e-12, float(np.max(np.abs(delta)))))
        return (delta * scale).astype(float)

    @staticmethod
    def _inside_shape(point, polygon):
        # Standard ray crossing, valid for the concave outline of the T.
        inside = False
        x, y = float(point[0]), float(point[1])
        for i in range(len(polygon)):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % len(polygon)]
            if (y1 > y) != (y2 > y):
                cross_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < cross_x:
                    inside = not inside
        return inside

    def _navigate(self, robot, target, block, rot, width, len_h, len_v, radius):
        """Shortest small visibility-graph route around an inflated T hull."""
        def hull(pad):
            top = width * 0.5 + pad
            bar_bottom = -width * 0.5 - pad
            half_h = len_h * 0.5 + pad
            half_stem = width * 0.5 + pad
            bottom = -len_v - width * 0.5 - pad
            local = np.array([[-half_h, top], [half_h, top],
                              [half_h, bar_bottom], [half_stem, bar_bottom],
                              [half_stem, bottom], [-half_stem, bottom],
                              [-half_stem, bar_bottom], [-half_h, bar_bottom]])
            return block + local @ rot.T

        obstacle = hull(radius + 0.055)
        # Wide clearance is important around the long stem tip; otherwise a
        # nominal visibility vertex becomes an accidental lever arm.
        waypoints = hull(radius + (0.20 if self.wide_clearance else 0.15))
        nodes = [robot, target] + [p for p in waypoints]

        def clear(a, b):
            # Dense sampling is reliable here (six edges, paths under 5 m)
            # and much cheaper than physics-based planning.
            n = min(40, max(2, int(np.linalg.norm(b - a) / 0.04)))
            for frac in np.linspace(0.03, 0.97, n):
                if self._inside_shape(a + frac * (b - a), obstacle):
                    return False
            return True

        count = len(nodes)
        dist = [float("inf")] * count
        prev = [-1] * count
        dist[0] = 0.0
        unseen = set(range(count))
        while unseen:
            u = min(unseen, key=lambda k: dist[k])
            unseen.remove(u)
            if not math.isfinite(dist[u]) or u == 1:
                break
            for v in list(unseen):
                if clear(nodes[u], nodes[v]):
                    nd = dist[u] + float(np.linalg.norm(nodes[v] - nodes[u]))
                    if nd < dist[v]:
                        dist[v], prev[v] = nd, u
        if not math.isfinite(dist[1]):
            return self._move(robot, target)
        node = 1
        while prev[node] not in (-1, 0):
            node = prev[node]
        return self._move(robot, nodes[node])

    def get_action(self, state):
        s = np.asarray(state, dtype=float)
        block, robot, goal = s[0:2], s[16:18], s[29:31]
        theta, goal_theta = float(s[2]), float(s[31])
        length_h = float(s[13])
        width, length_v = float(s[12]), float(s[14])
        radius = float(s[28])
        pos_delta = goal - block
        pos_err = float(np.linalg.norm(pos_delta))
        ang_err = _wrap(goal_theta - theta)
        rot = self._rot(theta)
        angle_limit = 0.135 if self.rotation_bursts >= 12 else 0.110

        # Conservative axis-aligned clearance of the currently oriented T.
        verts_local = np.array([[-length_h * 0.5, width * 0.5],
                                [length_h * 0.5, width * 0.5],
                                [length_h * 0.5, -width * 0.5],
                                [width * 0.5, -length_v - width * 0.5],
                                [-width * 0.5, -length_v - width * 0.5],
                                [-length_h * 0.5, -width * 0.5]])
        verts_world = block + verts_local @ rot.T
        wall_clearance = min(float(np.min(verts_world)),
                             float(5.0 - np.max(verts_world)))

        if self.phase.endswith("retreat"):
            action = self._move(robot, robot + self.retreat_vec * 0.30)
            self.push_steps += 1
            if self.push_steps >= 5:
                if self.phase == "rotate_retreat" and self.alternate_rotation:
                    self.normal *= -1.0
                self.phase = "decide"
            self.step += 1
            return action

        if self.phase == "decide":
            # Rotation near a wall is hazardous because every torque stroke
            # also translates the body. First push toward its (valid) goal.
            if (self.allow_wall_escape and wall_clearance < 0.32 and
                    pos_err > 0.20):
                self.escape_push = True
                self.phase = "translate_stage"
                self.stage_steps = 0
            elif abs(ang_err) > angle_limit:
                self.escape_push = False
                torque_sign = 1.0 if ang_err > 0.0 else -1.0
                use_nearest = (self.prefer_nearest or
                               (self.allow_fine_nearest and abs(ang_err) <= 0.35))
                self.alternate_rotation = not use_nearest
                if self.alternate_rotation:
                    self.side = -torque_sign * self.normal
                else:
                    choices = []
                    for normal in (1.0, -1.0):
                        side = -torque_sign * normal
                        pt = block + rot @ np.array([side * 0.40 * length_h,
                                                      normal * 0.33])
                        penalty = 20.0 if np.any(pt < 0.105) or np.any(pt > 4.895) else 0.0
                        choices.append((float(np.linalg.norm(robot - pt)) + penalty,
                                        normal, side))
                    _, self.normal, self.side = min(choices, key=lambda item: item[0])
                if abs(ang_err) < 0.20:
                    self.rotate_stroke = 5
                elif abs(ang_err) > 0.60:
                    self.rotate_stroke = 10
                else:
                    self.rotate_stroke = 7
                self.phase = "rotate_stage"
                self.stage_steps = 0
            elif pos_err > 0.018:
                self.escape_push = False
                self.phase = "translate_stage"
                self.stage_steps = 0
            else:
                self.step += 1
                return np.zeros(2, dtype=float)

        if self.phase.startswith("rotate"):
            # On the upper face force points down; on the lower face it points
            # up. Choosing the opposite cap ends gives the same torque sign.
            xoff = self.side * 0.40 * length_h
            if self.phase == "rotate_stage":
                self.stage_steps += 1
                if self.stage_steps > 80:
                    # A wall can invalidate the preferred side of the paired
                    # maneuver. The opposite face creates identical torque.
                    self.normal *= -1.0
                    self.side *= -1.0
                    self.stage_steps = 0
                    xoff = self.side * 0.40 * length_h
                target = block + rot @ np.array([xoff, self.normal * 0.33])
                if np.linalg.norm(robot - target) < 0.035:
                    self.phase = "rotate_push"
                    self.push_steps = 0
            contacting = self.phase == "rotate_push"
            if self.phase == "rotate_push":
                target = block + rot @ np.array([xoff, -self.normal * 0.05])
                self.push_steps += 1
                # The first few commands close the air gap; the last four
                # provide the useful crossbar contact stroke.
                if self.push_steps >= self.rotate_stroke:
                    self.rotation_bursts += 1
                    self.retreat_vec = rot @ np.array([0.0, self.normal])
                    self.push_steps = 0
                    self.phase = "rotate_retreat"
            if contacting:
                action = self._move(robot, target)
            else:
                action = self._navigate(robot, target, block, rot, width,
                                        length_h, length_v, radius)
        else:
            direction = pos_delta / pos_err if pos_err > 1e-9 else np.array([1.0, 0.0])
            # The T pose is at its cap/stem junction, not at its centre of
            # mass. A force line through this lower point is torque-neutral.
            push_center = block + rot @ np.array([0.0, -self.com_fraction * length_v])
            if self.phase == "translate_stage":
                self.stage_steps += 1
                # Stand beyond the longest possible stem before approaching
                # the pose origin along the desired force line.
                stand_off = 1.38 if self.prefer_nearest else 1.12
                feasible = stand_off
                for j in range(2):
                    if direction[j] > 1e-9:
                        feasible = min(feasible, (push_center[j] - 0.105) / direction[j])
                    elif direction[j] < -1e-9:
                        feasible = min(feasible, (4.895 - push_center[j]) / -direction[j])
                target = push_center - direction * max(0.0, feasible)
                if np.linalg.norm(robot - target) < 0.040 or self.stage_steps > 100:
                    self.phase = "translate_push"
                    self.push_steps = 0
                    self.push_origin = block.copy()
                    self.push_goal_dist = pos_err
            contacting = self.phase == "translate_push"
            if self.phase == "translate_push":
                target = push_center + direction * 0.20
                self.push_steps += 1
                moved = float(np.linalg.norm(block - self.push_origin))
                # A long stroke is necessary: much of it merely traverses the
                # stand-off gap. Replan after useful object motion, not after a
                # fixed handful of empty-space commands.
                travel = 0.45
                bad_angle = (not self.escape_push and
                             abs(ang_err) > max(0.125, angle_limit))
                if (moved >= min(travel, self.push_goal_dist + 0.01) or pos_err < 0.025 or
                        self.push_steps >= 40 or bad_angle):
                    self.retreat_vec = -direction
                    self.push_steps = 0
                    self.phase = "translate_retreat"
            if contacting:
                action = self._move(robot, target)
            else:
                action = self._navigate(robot, target, block, rot, width,
                                        length_h, length_v, radius)

        self.step += 1
        return action
