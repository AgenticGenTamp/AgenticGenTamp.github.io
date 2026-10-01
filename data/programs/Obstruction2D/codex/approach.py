"""Fast staged manipulation policy for the variable-count Obstruction2D task."""

import math
import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.space = observation_space
        self.types = {n: observation_space.get_type(n) for n in
                      ("crv_robot", "target_block", "target_surface")}

    @staticmethod
    def _clip(x, limit):
        return max(-limit, min(limit, x))

    @staticmethod
    def _angle(a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def reset(self, state, info):
        obstacles = sorted(n for n in state.get_object_names()
                           if n.startswith("obstruction"))
        surface = state.get_objects(self.types["target_surface"])[0]
        sx = state.get(surface, "x")
        sw = state.get(surface, "width")
        block = state.get_objects(self.types["target_block"])[0]
        bx, bw = state.get(block, "x"), state.get(block, "width")
        desired_block_x = sx + (sw - bw) / 2.0
        self.jobs = []
        self.pick_offsets = {}
        # Keep parked objects out of the entire future block transport corridor.
        occupied = [(min(bx, sx), max(bx + bw, sx + sw))]
        reserved = [(bx, bx + bw), (sx, sx + sw)]
        movable_intervals = []
        for other_name in obstacles + [block.name]:
            other = state.get_object_from_name(other_name)
            ax = state.get(other, "x")
            movable_intervals.append((other_name, ax,
                                      ax + state.get(other, "width")))
        for name in obstacles:
            obj = state.get_object_from_name(name)
            width = state.get(obj, "width")
            x = state.get(obj, "x")
            # Objects merely adjacent to the platform need not be disturbed.
            near_goal = not (x >= desired_block_x + bw + 0.035 or
                             x + width <= desired_block_x - 0.035)
            near_pickup = not (x >= bx + bw + 0.035 or
                               x + width <= bx - 0.035)
            if not (near_goal or near_pickup):
                continue
            center = x + width / 2.0
            # Favor the exposed outer edge when the platform blocks a central
            # approach. The vacuum pad tolerates this off-center grasp.
            inset = min(0.02, 0.2 * width)
            candidates_pick = [x + inset, center, x + width - inset]
            def clearance(point):
                distances = []
                for other_name, a, b in movable_intervals:
                    if other_name == name:
                        continue
                    if point < a:
                        distances.append(a - point)
                    elif point > b:
                        distances.append(point - b)
                    else:
                        distances.append(-min(point - a, b - point))
                return min(distances) if distances else 10.0
            pick_x = max(candidates_pick, key=clearance)
            pick_x = max(0.11, min(1.50, pick_x))
            offset = pick_x - center
            self.pick_offsets[name] = offset
            # Select a free storage interval from a fixed workspace-independent
            # grid; this scales by scanning rather than relying on object indices.
            candidates = [0.14 + 0.11 * i for i in range(13)]
            candidates.sort(key=lambda c: -abs(c - (sx + sw / 2.0)))
            drop_x = candidates[0]
            for candidate in candidates:
                lo = candidate - (width / 2.0 + offset)
                hi = lo + width
                if (lo >= 0.005 and hi <= 1.595 and
                        all(hi + 0.015 <= a or lo - 0.015 >= b
                            for a, b in occupied)):
                    drop_x = candidate
                    break
            else:
                # If the pickup-to-goal corridor spans almost the whole table,
                # ground objects in separated interior gaps instead.
                for candidate in candidates:
                    lo = candidate - (width / 2.0 + offset)
                    hi = lo + width
                    if (lo >= 0.005 and hi <= 1.595 and
                            all(hi + 0.025 <= a or lo - 0.025 >= b
                                for a, b in reserved)):
                        drop_x = candidate
                        break
            drop_x = max(width / 2.0 + offset + 0.005,
                         min(1.595 - width / 2.0 + offset, drop_x))
            lo = drop_x - (width / 2.0 + offset)
            occupied.append((lo, lo + width))
            reserved.append((lo, lo + width))
            self.jobs.append((name, drop_x, 0.1, False))
        surface_center = sx + sw / 2.0
        block_center = bx + bw / 2.0
        # An off-center grasp lets the arm place blocks on surfaces close to a
        # workspace boundary without requiring the base to leave the table.
        low_offset = max(-bw / 2.0 - 0.03,
                         0.11 - block_center, 0.11 - surface_center)
        high_offset = min(bw / 2.0 + 0.03,
                          1.50 - block_center, 1.50 - surface_center)
        self.pick_offsets[block.name] = max(low_offset,
                                            min(high_offset, 0.0))
        self.jobs.append((block.name, desired_block_x, 0.1, True))
        self.job_index = 0
        self.stage = "lift_safe"
        self.stable = 0
        self.stage_age = 0

    def _objects(self, state):
        robot = state.get_objects(self.types["crv_robot"])[0]
        name, dx, dy, final = self.jobs[self.job_index]
        return robot, state.get_object_from_name(name), dx, dy, final

    def _command(self, state, xy=None, theta=None, arm=None, vacuum=0.0):
        robot = state.get_objects(self.types["crv_robot"])[0]
        get = state.get
        a = np.zeros(5, dtype=np.float32)
        if xy is not None:
            a[0] = self._clip(xy[0] - get(robot, "x"), 0.05)
            a[1] = self._clip(xy[1] - get(robot, "y"), 0.05)
        if theta is not None:
            a[2] = self._clip(self._angle(theta - get(robot, "theta")), 0.1963495)
        if arm is not None:
            a[3] = self._clip(arm - get(robot, "arm_joint"), 0.1)
        a[4] = vacuum
        return a

    def _reached(self, state, xy=None, theta=None, arm=None, tol=0.012):
        robot = state.get_objects(self.types["crv_robot"])[0]
        get = state.get
        ok = True
        if xy is not None:
            ok &= math.hypot(get(robot, "x") - xy[0],
                             get(robot, "y") - xy[1]) < tol
        if theta is not None:
            ok &= abs(self._angle(get(robot, "theta") - theta)) < 0.025
        if arm is not None:
            ok &= abs(get(robot, "arm_joint") - arm) < tol
        return ok

    def get_action(self, state):
        robot, obj, drop_x, drop_y, final = self._objects(state)
        ox = (state.get(obj, "x") + state.get(obj, "width") / 2.0
              + self.pick_offsets[obj.name])
        down = -math.pi / 2.0
        transit_y = 0.72
        work_y = 0.40

        if self.stage == "lift_safe":
            if self._reached(state, arm=0.1): self.stage = "turn_pick"
            return self._command(state, arm=0.1)
        if self.stage == "turn_pick":
            if self._reached(state, theta=down): self.stage = "rise_pick"
            return self._command(state, theta=down, arm=0.1)
        if self.stage == "rise_pick":
            target = (state.get(robot, "x"), transit_y)
            if abs(state.get(robot, "y") - transit_y) < 0.012: self.stage = "across_pick"
            return self._command(state, xy=target, theta=down, arm=0.1)
        if self.stage == "across_pick":
            target = (ox, transit_y)
            if self._reached(state, xy=target):
                self.stage, self.stage_age = "down_pick", 0
            return self._command(state, xy=target, theta=down, arm=0.1)
        if self.stage == "down_pick":
            # Descend with suction already active. Collision resolution stops the
            # gripper at contact; a coarse downward step reliably establishes it.
            self.stage_age += 1
            if self.stage_age >= 25:
                self.stage = "retract_carry"
            ry = state.get(robot, "y")
            contact_y = (state.get(obj, "y") + state.get(obj, "height")
                         + 0.22)
            next_y = 0.25 if ry > contact_y + 0.06 else ry - 0.005
            return self._command(state, xy=(ox, next_y), theta=down,
                                 arm=0.2, vacuum=1.0)
        if self.stage == "extend_pick":
            if self._reached(state, arm=0.2, tol=0.018):
                self.stage, self.stable = "grasp", 0
            return self._command(state, theta=down, arm=0.2)
        if self.stage == "grasp":
            self.stable += 1
            if self.stable >= 2: self.stage = "retract_carry"
            return self._command(state, theta=down, arm=0.2, vacuum=1.0)
        if self.stage == "retract_carry":
            if self._reached(state, arm=0.1): self.stage = "rise_carry"
            return self._command(state, theta=down, arm=0.1, vacuum=1.0)
        if self.stage == "rise_carry":
            target = (state.get(robot, "x"), transit_y)
            if abs(state.get(robot, "y") - transit_y) < 0.012: self.stage = "across_drop"
            return self._command(state, xy=target, theta=down, arm=0.1, vacuum=1.0)
        if self.stage == "across_drop":
            if final:
                error = drop_x - state.get(obj, "x")
                target = (state.get(robot, "x") + error, transit_y)
                if abs(error) < 0.008:
                    self.stage, self.stage_age = "final_down", 0
            else:
                target = (drop_x, transit_y)
                if self._reached(state, xy=target):
                    self.stage, self.stage_age = "park_down", 0
            return self._command(state, xy=target, theta=down, arm=0.1, vacuum=1.0)
        if self.stage == "park_down":
            self.stage_age += 1
            if self.stage_age >= 40:
                self.stage, self.stable = "release", 0
            target = (state.get(robot, "x"), state.get(robot, "y") - 0.01)
            next_arm = min(0.2, state.get(robot, "arm_joint") + 0.005)
            return self._command(state, xy=target, theta=down, arm=next_arm,
                                 vacuum=1.0)
        if self.stage == "final_down":
            # The success predicate fires while the block is still held. Fine
            # descent avoids atomic rejection against the static platform.
            target = (state.get(robot, "x"), state.get(robot, "y") - 0.01)
            next_arm = min(0.2, state.get(robot, "arm_joint") + 0.005)
            return self._command(state, xy=target, theta=down, arm=next_arm,
                                 vacuum=1.0)
        if self.stage == "down_drop":
            target = (drop_x, work_y)
            if self._reached(state, xy=target): self.stage = "extend_drop"
            return self._command(state, xy=target, theta=down, arm=0.1, vacuum=1.0)
        if self.stage == "extend_drop":
            if self._reached(state, arm=0.2, tol=0.018):
                self.stage, self.stable = "release", 0
            return self._command(state, theta=down, arm=0.2, vacuum=1.0)

        self.stable += 1
        if self.stable >= 2 and not final:
            self.job_index += 1
            self.stage, self.stable = "lift_safe", 0
        return self._command(state, theta=down, arm=0.2, vacuum=0.0)
