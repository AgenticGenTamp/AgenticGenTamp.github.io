"""Reactive controller for StickButton2DEnv."""

import math
import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.circle_type = observation_space.get_type("circle")
        self.low = np.asarray(action_space.low, dtype=float)
        self.high = np.asarray(action_space.high, dtype=float)
        self.phase = "direct"
        self.previous_robot = None
        self.last_move = (0.0, 0.0)
        self.detour_y = None
        self.cross_x = None
        self.counter = 0
        self.approach_speed = 0.05
        self.approach_side = 1.0
        self.rotate_goal = 0.0

    def reset(self, state, info):
        self.phase = "direct"
        robot = state.get_object_from_name("robot")
        self.previous_robot = (float(state.get(robot, "x")),
                               float(state.get(robot, "y")))
        self.last_move = (0.0, 0.0)
        self.detour_y = None
        self.cross_x = None
        self.counter = 0
        self.approach_speed = 0.05
        self.approach_side = 1.0
        self.rotate_goal = 0.0

    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def get_action(self, state):
        robot = state.get_object_from_name("robot")
        rx = float(state.get(robot, "x"))
        ry = float(state.get(robot, "y"))
        rt = float(state.get(robot, "theta"))
        arm = float(state.get(robot, "arm_joint"))
        action = np.zeros(5, dtype=float)
        # The base presses buttons directly.  Red circles are the remaining
        # goals; this enumeration works for arbitrary object counts.
        buttons = list(state.get_objects(self.circle_type))
        active = [b for b in buttons if float(state.get(b, "color_r")) >
                  float(state.get(b, "color_g"))]
        if not active:
            return action.astype(self.action_space.dtype)
        target = min(active, key=lambda b: max(abs(float(state.get(b, "x")) - rx),
                                                abs(float(state.get(b, "y")) - ry)))
        if self.phase in ("sweep", "rotate_stick"):
            # Use the held stick on highest goals first; leave floor goals for
            # the base only after the elevated workspace is clear.
            target = max(active, key=lambda b: float(state.get(b, "y")))
        elif self.phase == "release_low":
            target = min(active, key=lambda b: float(state.get(b, "y")))
        tx, ty = (float(state.get(target, f)) for f in ("x", "y"))

        moved = math.hypot(rx - self.previous_robot[0], ry - self.previous_robot[1])
        commanded = max(abs(self.last_move[0]), abs(self.last_move[1]))
        stick = state.get_object_from_name("stick")
        sx, sy = (float(state.get(stick, f)) for f in ("x", "y"))

        if self.phase == "direct":
            # Clear the lower workspace efficiently before picking up the stick.
            low_targets = [b for b in active if float(state.get(b, "y")) < 1.35]
            if not low_targets or (commanded > 0.015 and moved < 0.004):
                self.phase = "stage"
                self.counter = 0
                if sx > 2.75:
                    self.approach_side = -1.0
                elif sx < 0.60:
                    self.approach_side = 1.0
                else:
                    self.approach_side = -1.0 if rx < sx else 1.0
            else:
                target = min(low_targets,
                    key=lambda b: max(abs(float(state.get(b, "x")) - rx),
                                      abs(float(state.get(b, "y")) - ry)))
                tx, ty = (float(state.get(target, f)) for f in ("x", "y"))
                aim = math.atan2(ty - ry, tx - rx)
                # Axis-separated travel prevents a workspace-bound y request
                # from rejecting useful horizontal progress as well.
                if abs(tx - rx) > 0.025:
                    action[0] = tx - rx
                else:
                    action[1] = ty - ry
                action[2] = self._wrap(aim - rt)
                action[3] = 0.2 - arm

        if self.phase == "stage":
            # Approach from a shallow angle so the vacuum pad, rather than the
            # arm shaft or base, is the first geometry to contact the stick.
            gx, gy = sx + self.approach_side * 0.60, sy - 0.16
            if abs(gy - ry) > 0.012:
                action[1] = gy - ry
            else:
                action[0] = gx - rx
            action[3] = 0.1 - arm
            if abs(gx - rx) < 0.015 and abs(gy - ry) < 0.015:
                self.phase = "acquire"
                self.last_move = (0.0, 0.0)
                self.approach_speed = 0.05

        elif self.phase == "acquire":
            # Empirically the suction face is centered 0.263 from the base.
            aim = math.atan2(0.16, -self.approach_side * 0.60)
            err = self._wrap(aim - rt)
            action[2] = err
            action[3] = 0.2 - arm
            gx = sx - 0.263 * math.cos(aim)
            gy = sy - 0.263 * math.sin(aim)
            if abs(err) < 0.07 and arm > 0.19:
                dx, dy = gx - rx, gy - ry
                scale = min(1.0, self.approach_speed /
                            max(math.hypot(dx, dy), 1e-9))
                action[0], action[1] = dx * scale, dy * scale
            if math.hypot(gx - rx, gy - ry) < 0.003 and abs(err) < 0.02:
                self.counter += 1
            else:
                self.counter = 0
            if self.counter >= 1:
                self.phase = "attach"
                self.counter = 0

        elif self.phase == "attach":
            # Vacuum must be toggled in a stationary contact configuration;
            # collision-rejected motion would also reject the vacuum update.
            action[4] = 1.0
            self.counter += 1
            if self.counter >= 2:
                self.phase = "pull"
                self.counter = 0

        elif self.phase == "pull":
            action[0] = 0.05
            action[4] = 1.0
            self.counter += 1
            if self.counter >= 10:
                self.phase = "sweep"

        elif self.phase == "sweep":
            action[4] = 1.0
            if ty < 0.70:
                self.phase = "release_low"
                action[4] = 0.0
            else:
                # Rectangle pose is its first endpoint; theta=0 extends upward.
                st = float(state.get(stick, "theta"))
                length = float(state.get(stick, "height"))
                vx, vy = -math.sin(st), math.cos(st)
                proj = max(0.0, min(length, (tx - sx) * vx + (ty - sy) * vy))
                px, py = sx + proj * vx, sy + proj * vy
                # As elsewhere, avoid losing horizontal progress when a tiny
                # vertical component is rejected by the workspace boundary.
                edge = tx < 0.35 or tx > 3.15
                vertical_feasible = ty < 1.80 or ty < py
                if edge and vertical_feasible and abs(ty - py) > 0.025:
                    action[1] = ty - py
                elif abs(tx - px) > 0.025:
                    action[0] = tx - px
                else:
                    action[1] = ty - py
                if commanded > 0.015 and moved < 0.004:
                    # At a workspace limit, tilt the long stick toward the goal.
                    if ty > 2.15:
                        if tx > 3.15:
                            self.rotate_goal = -0.20
                        elif tx < 0.35:
                            self.rotate_goal = 0.20
                        else:
                            self.rotate_goal = 0.0
                    else:
                        self.rotate_goal = math.atan2(-(tx - sx), ty - sy)
                        self.rotate_goal = max(-1.2, min(1.2, self.rotate_goal))
                    self.phase = "rotate_stick"
                    action[0] = action[1] = 0.0

        elif self.phase == "release_low":
            # Very low edge buttons are simpler to finish with the base after
            # dropping the stick, which cannot rotate below the floor.
            if ty >= 0.70:
                self.phase = "stage"
                self.approach_side = -1.0 if rx < sx else 1.0
                action[3] = 0.1 - arm
            else:
                aim = math.atan2(ty - ry, tx - rx)
                action[2] = self._wrap(aim - rt)
                action[3] = 0.2 - arm
                if (rx - sx) * (tx - sx) < 0.0 and ry > sy - 0.16:
                    action[1] = (sy - 0.16) - ry
                    action[2] = self._wrap(-math.pi / 2.0 - rt)
                    action[3] = 0.1 - arm
                elif abs(tx - rx) > 0.025:
                    action[0] = tx - rx
                else:
                    action[1] = ty - ry

        elif self.phase == "rotate_stick":
            action[4] = 1.0
            st = float(state.get(stick, "theta"))
            action[2] = self._wrap(self.rotate_goal - st)
            # First make room for the rigid grasp point to arc during rotation.
            if sx > 2.50:
                action[0] = -0.05
                action[2] = 0.0
            elif sx < 1.00:
                action[0] = 0.05
                action[2] = 0.0
            elif ry < 0.40:
                action[1] = 0.05
                action[2] = 0.0
            elif ry > 0.80:
                action[1] = -0.05
                action[2] = 0.0
            if abs(self._wrap(self.rotate_goal - st)) < 0.015:
                self.phase = "sweep"

        clipped = np.clip(action, self.low, self.high)
        self.previous_robot = (rx, ry)
        self.last_move = (float(clipped[0]), float(clipped[1]))
        return clipped.astype(self.action_space.dtype)
