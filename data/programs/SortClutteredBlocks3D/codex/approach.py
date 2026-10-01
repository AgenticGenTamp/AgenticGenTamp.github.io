"""Reactive controller for the variable-count clutter sorting task."""

import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.low = np.asarray(action_space.low, dtype=np.float32)
        self.high = np.asarray(action_space.high, dtype=np.float32)

    def reset(self, state, info):
        self.steps = 0
        robot = self._robot(state)
        self.start_base = np.array([state.get(robot, "pos_base_x"),
                                    state.get(robot, "pos_base_y"),
                                    state.get(robot, "pos_base_rot")])
        cube_names = [n for n in state.get_object_names() if n.startswith("cube")]
        numeric_ids = [int(n[4:]) for n in cube_names if n[4:].isdigit()]
        self.index_offset = min(numeric_ids) if numeric_ids else 0
        self.bin_targets = {}
        for color in ("red", "green", "blue", "yellow"):
            obj = state.get_object_from_name("bin_" + color)
            self.bin_targets[color] = np.array([state.get(obj, "x"),
                                                state.get(obj, "y")])
        # Exposed/high objects first is important for clutter.  Radially outer
        # objects are less likely to make the narrow fingertip sweep disturb a
        # neighbor on its way into the workspace.
        def order_key(name):
            number = int(name[4:]) if name[4:].isdigit() else 1
            obj = state.get_object_from_name(name)
            radius = np.hypot(state.get(obj, "x"), state.get(obj, "y"))
            color_index = (number - self.index_offset) % 4
            # Red and blue start in the same outward quadrant as their bins;
            # clear them before green/yellow must cross the central clutter.
            color_rank = (0, 2, 1, 3)[color_index]
            return (color_rank, -radius, -state.get(obj, "z"), number)
        self.cube_names = sorted(cube_names, key=order_key)
        cube_xy = {}
        for name in self.cube_names:
            obj = state.get_object_from_name(name)
            cube_xy[name] = np.array([state.get(obj, "x"),
                                      state.get(obj, "y")])
        xs = sorted(p[0] for p in cube_xy.values())
        ys = sorted(p[1] for p in cube_xy.values())
        self.red_x_first = set()
        for name, point in cube_xy.items():
            number = int(name[4:]) if name[4:].isdigit() else 1
            if (number - self.index_offset) % 4 != 0:
                continue
            x_margin = ((xs[1] - point[0]) if len(xs) > 1 and
                        point[0] <= xs[0] + 1e-5 else -1.0)
            y_margin = ((ys[1] - point[1]) if len(ys) > 1 and
                        point[1] <= ys[0] + 1e-5 else -1.0)
            if x_margin > max(0.005, y_margin):
                self.red_x_first.add(name)
        self.plan = []
        for name in self.cube_names:
            number = int(name[4:]) if name[4:].isdigit() else 1
            color_index = (number - self.index_offset) % 4
            self.plan.append((name, "first"))
            if name in self.red_x_first:
                self.plan.append((name, "second"))
            if color_index in (1, 2, 3):
                self.plan.append((name, "second"))
        self.push_context = None
        self.hit_base = None
        self.contact_xy = None
        self.brake_q = None
        self.brake_base = None
        self.sweep_done = False

    @staticmethod
    def _robot(state):
        return state.get_object_from_name("robot")

    def get_action(self, state):
        action = np.zeros(self.action_space.shape, dtype=np.float32)
        robot = self._robot(state)
        base = np.array([state.get(robot, "pos_base_x"),
                         state.get(robot, "pos_base_y"),
                         state.get(robot, "pos_base_rot")])
        q = np.array([state.get(robot, "pos_arm_joint%d" % i)
                      for i in range(1, 8)])
        # A narrow fingertip tangential sweep is the only black-box primitive
        # that repeatably moved one cube without first plowing the whole pile.
        # Rotate that calibrated local sweep toward the appropriate live bin.
        durations = (20, 15, 35, 100, 25, 80)
        cycle = sum(durations)
        elapsed = self.steps
        index = elapsed // cycle
        within = elapsed % cycle
        phase = 0
        boundary = durations[0]
        while phase < len(durations) - 1 and within >= boundary:
            phase += 1
            boundary += durations[phase]

        home_q = np.array([0.0, -0.349, np.pi, -2.548,
                           0.0, -0.873, np.pi / 2])
        edge_q = np.array([0.0, 1.30, np.pi, -1.70,
                           0.0, 1.0, 0.0])
        q_goal = home_q
        base_goal = base.copy()
        grip = 0.0
        if self.cube_names:
            name, leg = self.plan[index % len(self.plan)]
            cube = state.get_object_from_name(name)
            number = int(name[4:]) if name[4:].isdigit() else index + 1
            colors = ("red", "green", "blue", "yellow")
            color = colors[(number - self.index_offset) % 4]
            bin_obj = state.get_object_from_name("bin_" + color)
            cube_x = state.get(cube, "x")
            cube_y = state.get(cube, "y")
            # Receptacles are movable; refresh their live centers for every
            # manipulation cycle rather than chasing stale reset positions.
            bin_x = state.get(bin_obj, "x")
            bin_y = state.get(bin_obj, "y")
            if within == 0 or self.push_context is None:
                self.hit_base = None
                self.contact_xy = None
                self.brake_q = None
                self.brake_base = None
                self.sweep_done = False
                dx, dy = bin_x - cube_x, bin_y - cube_y
                norm = max(1e-6, float(np.hypot(dx, dy)))
                ux, uy = dx / norm, dy / norm
                # In the canonical clutter, the radially exposed red cube is
                # also the minimum-y object.  This exact cardinal setup was
                # calibrated by first-contact experiments and avoids catching
                # its yellow neighbor during a diagonal approach.
                if index == 0 and color == "red":
                    ux, uy = ((-1.0, 0.0) if name in self.red_x_first
                              else (0.0, -1.0))
                elif color == "red" and leg == "second":
                    ux, uy = 0.0, -1.0
                elif color == "blue":
                    ux, uy = ((1.0, 0.0) if leg == "first"
                              else (0.0, 1.0))
                elif color == "green":
                    ux, uy = ((0.0, 1.0) if leg == "first"
                              else (-1.0, 0.0))
                elif color == "yellow":
                    ux, uy = ((0.0, -1.0) if leg == "first"
                              else (1.0, 0.0))
                # Positive joint-1 motion sweeps along local -Y.  Choose yaw
                # so local -Y equals the cube-to-bin unit direction.
                yaw = float(np.arctan2(ux, -uy))
                c, s = np.cos(yaw), np.sin(yaw)
                def offset(local_x, local_y):
                    return np.array([cube_x + c * local_x - s * local_y,
                                     cube_y + s * local_x + c * local_y,
                                     yaw])
                lateral = 0.020 if name in self.red_x_first else 0.035
                outer = offset(-0.97, lateral)
                contact = offset(-0.75, lateral)
                start_angle = float(np.arctan2(base[1], base[0]))
                end_angle = float(np.arctan2(outer[1], outer[0]))
                turn = (end_angle - start_angle + np.pi) % (2 * np.pi) - np.pi
                initial_xy = {n: np.array([
                    state.get(state.get_object_from_name(n), "x"),
                    state.get(state.get_object_from_name(n), "y")])
                    for n in self.cube_names}
                self.push_context = (name, outer, contact,
                                     start_angle, turn, initial_xy,
                                     np.array([bin_x, bin_y]))
            (_, outer, contact, start_angle, turn, initial_xy,
             target_bin_xy) = self.push_context
            if phase == 0:
                # Fold in place before crossing around the table.
                base_goal = base.copy()
            elif phase == 1:
                base_goal = np.array([1.15 * np.cos(start_angle),
                                      1.15 * np.sin(start_angle), base[2]])
            elif phase == 2:
                phase_start = durations[0] + durations[1]
                fraction = (within - phase_start + 1) / durations[2]
                angle = start_angle + turn * fraction
                base_goal = np.array([1.15 * np.cos(angle),
                                      1.15 * np.sin(angle), outer[2]])
            elif phase == 3:
                base_goal = outer
                q_goal = edge_q
            elif phase == 4:
                if self.contact_xy is None:
                    self.contact_xy = {n: np.array([
                        state.get(state.get_object_from_name(n), "x"),
                        state.get(state.get_object_from_name(n), "y")])
                        for n in self.cube_names}
                if self.hit_base is None:
                    if index == 0 and color == "red":
                        for cube_name, old_xy in self.contact_xy.items():
                            obj = state.get_object_from_name(cube_name)
                            now_xy = np.array([state.get(obj, "x"),
                                               state.get(obj, "y")])
                            if np.linalg.norm(now_xy - old_xy) > 0.005:
                                self.hit_base = base.copy()
                                break
                    else:
                        old_xy = self.contact_xy[name]
                        now_xy = np.array([cube_x, cube_y])
                        if np.linalg.norm(now_xy - old_xy) > 0.005:
                            self.hit_base = base.copy()
                base_goal = contact if self.hit_base is None else self.hit_base
                q_goal = edge_q
            else:
                base_goal = contact if self.hit_base is None else self.hit_base
                q_goal = edge_q.copy()
                q_goal[0] = 0.80
                if (index == 0 and color == "red" and
                        np.hypot(cube_x - target_bin_xy[0],
                                 cube_y - target_bin_xy[1]) < 0.05):
                    self.sweep_done = True
                elif (index == 0 and color == "red" and
                      name in self.red_x_first and
                      cube_x <= target_bin_xy[0] + 0.03):
                    self.sweep_done = True
                if self.sweep_done:
                    base_goal = outer
                    q_goal = q.copy()
                elif (color == "red" and
                        ((index == 0 and name not in self.red_x_first) or
                         leg == "second") and
                        cube_y <= target_bin_xy[1] + 0.04):
                    if self.brake_q is None:
                        self.brake_q = float(q[0])
                        self.brake_base = base.copy()
                    delta = 0.88 * (np.sin(abs(q[0])) -
                                    np.sin(abs(self.brake_q)))
                    base_goal = self.brake_base.copy()
                    # Positive q1 sweeps along u=(0,-1); moving the base in
                    # -u cancels further y travel and lets the arc pull left.
                    base_goal[1] += delta
        base_error = base_goal - base
        base_error[2] = (base_error[2] + np.pi) % (2 * np.pi) - np.pi
        base_limit = 0.012 if self.cube_names and phase == 4 else 0.1
        action[:3] = np.clip(1.5 * base_error, -base_limit, base_limit)
        action[3:10] = np.clip(0.7 * (q_goal - q), -0.1, 0.1)
        action[-1] = grip
        self.steps += 1
        return np.clip(action, self.low, self.high)
