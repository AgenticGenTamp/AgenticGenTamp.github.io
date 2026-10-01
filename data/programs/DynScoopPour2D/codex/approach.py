import math
import numpy as np


class GeneratedApproach:
    """Grasp the supplied L-hook and repeatedly scoop objects over the wall."""

    def __init__(self, action_space, observation_space, primitives):
        self.low = np.asarray(action_space.low, dtype=float)
        self.high = np.asarray(action_space.high, dtype=float)
        self.action_dtype = action_space.dtype
        self.reset(None, None)

    @staticmethod
    def _angle_error(x):
        return (x + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _get(state, name, feature):
        return float(state.get(state.get_object_from_name(name), feature))

    def reset(self, state, info):
        self.stage = 0
        self.wait = 0
        self.stage_age = 0
        self.hook_x = None
        self.cycles = 0
        self.low_y = 0.75
        self.left_x = 0.55
        self.sweep_x = 1.48
        self.fast_pickup = False
        self.regrasp_x = 1.4
        self.regrasp_y = 0.75
        self.small_count = 0
        self.best_right = 0
        self.last_progress_cycle = 0

    def _waypoint(self):
        hx = self.hook_x
        pickup_x = min(hx - 0.10 if self.small_count <= 10 else hx, 3.299)
        safe_x = min(pickup_x, 3.15)
        boundary_pickup = hx > 3.40
        pickup_theta = -1.40 if boundary_pickup else -math.pi / 2
        pickup_y = 0.95 if boundary_pickup else 0.75
        pickup_joint = 0.40 if boundary_pickup else 0.20
        # x, y, angle, arm joint, finger gap, settling steps
        initial = [
            (safe_x, pickup_y if self.fast_pickup and pickup_x <= 3.15 else 2.35,
             pickup_theta, pickup_joint, 0.25, 3),
            (safe_x, pickup_y, pickup_theta, pickup_joint, 0.25, 3),
            (pickup_x, pickup_y, pickup_theta, pickup_joint, 0.25, 3),
            (pickup_x, pickup_y, pickup_theta, pickup_joint, 0.08, 8),
            # Normalize the held-hook pose before transport; this also frees an
            # angled boundary pickup from the outer wall.
            (pickup_x, pickup_y, -math.pi / 2, 0.20, 0.08, 3),
        ]
        prep = [
            (self.left_x, 2.40, -math.pi / 2, 0.20, 0.08, 2),
            (self.left_x, self.low_y, -math.pi / 2, 0.20, 0.08, 2),
            (self.sweep_x, self.low_y, -math.pi / 2, 0.20, 0.08, 8),
        ]
        if self.stage < len(initial):
            return initial[self.stage]
        if self.stage < len(initial) + len(prep):
            return prep[self.stage - len(initial)]
        if self.small_count <= 10:
            slow_scoop = [
                (self.sweep_x, 2.40, -math.pi / 2, 0.20, 0.08, 2),
                (2.65, 2.40, -math.pi / 2, 0.20, 0.08, 6),
                (2.65, 2.40, 0.0, 0.20, 0.08, 4),
                (self.left_x, 2.40, -math.pi / 2, 0.20, 0.08, 2),
                (self.left_x, self.low_y, -math.pi / 2, 0.20, 0.08, 2),
                (self.sweep_x, self.low_y, -math.pi / 2, 0.20, 0.08, 6),
            ]
            return slow_scoop[(self.stage - 8) % len(slow_scoop)]
        # Fast floor-level paddle cycle.  Opening and regrasping after every throw
        # resets small pose slips in the dynamic hook; the back-and-forth sweep
        # presents a fresh layer of an arbitrarily large pile on each repeat.
        # Dense piles need the full wall-side reach on every attempt.  Retreating
        # after a missed launch makes subsequent throws strictly shorter.
        paddle_x = 1.48
        throw_y = self.low_y
        target_count = (self.small_count + 1) // 2
        dense_finish = (self.cycles % 2 == 1 and
                        0 < self.best_right < target_count)
        if dense_finish:
            # A floor flick sometimes launches only part of a dense pile.  Once
            # it has made real progress, retain the hook and use the slower but
            # dependable lift-over-wall motion to clear the remainder.
            finish = [
                (1.30, 2.40, -math.pi / 2, 0.20, 0.08, 2),
                (2.65, 2.40, -math.pi / 2, 0.20, 0.08, 6),
                (2.65, 2.40, 0.0, 0.20, 0.08, 4),
                (self.left_x, 2.40, -math.pi / 2, 0.20, 0.08, 2),
                (self.left_x, self.low_y, -math.pi / 2, 0.20, 0.08, 2),
                (self.sweep_x, self.low_y, -math.pi / 2, 0.20, 0.08, 6),
                (self.sweep_x, self.low_y, -math.pi / 2, 0.20, 0.08, 2),
            ]
            return finish[(self.stage - len(initial) - len(prep)) % len(finish)]
        paddle = [
            (paddle_x, throw_y, 0.0, 0.20, 0.08, 2),
            (1.40, self.low_y, -math.pi / 2, 0.20, 0.08, 2),
            (1.40, self.low_y, -math.pi / 2, 0.20, 0.25, 2),
            (self.regrasp_x, self.regrasp_y, -math.pi / 2, 0.20, 0.25, 2),
            (self.regrasp_x, self.regrasp_y, -math.pi / 2, 0.20, 0.08, 3),
            (0.25 if self.cycles == 0 else 0.55,
             0.75, -math.pi / 2, 0.20, 0.08, 2),
            (paddle_x, 0.75, -math.pi / 2, 0.20, 0.08, 2),
        ]
        return paddle[(self.stage - len(initial) - len(prep)) % len(paddle)]

    def get_action(self, state):
        if self.hook_x is None:
            self.hook_x = self._get(state, "hook", "x")
            self.fast_pickup = self._get(state, "robot", "x") > 1.95
            self.small_count = sum(n.startswith("small_") for n in state.get_object_names())

        right_now = 0
        for name in state.get_object_names():
            if name.startswith("small_") and self._get(state, name, "x") > 1.75:
                right_now += 1
        if right_now > self.best_right:
            self.best_right = right_now
            self.last_progress_cycle = self.cycles

        # Retry the pickup if contact displaced the hook without grasping it.
        if self.stage == 4 and self._get(state, "hook", "held") < 0.5:
            self.stage = 0
            self.wait = 0
            self.stage_age = 0
            self.hook_x = self._get(state, "hook", "x")

        # In the paddle loop, never spend a sweep cycle empty-handed.  If the
        # close missed, reopen and align locally with the resting hook.
        if self.small_count > 10 and self.stage >= 8:
            phase = (self.stage - 8) % 7
            if phase in (5, 6) and self._get(state, "hook", "held") < 0.5:
                cycle_base = self.stage - phase
                self.stage = cycle_base + 2
                self.regrasp_x = min(self._get(state, "hook", "x"), 1.48)
                self.wait = 0
                self.stage_age = 0

        x, y, theta, joint, gap, settle = self._waypoint()
        rx = self._get(state, "robot", "x")
        ry = self._get(state, "robot", "y")
        rt = self._get(state, "robot", "theta")
        rj = self._get(state, "robot", "arm_joint")
        rg = self._get(state, "robot", "finger_gap")
        errors = (x-rx, y-ry, self._angle_error(theta-rt), joint-rj, gap-rg)
        scales = (0.03, 0.03, 0.098, 0.08, 0.015)
        arrived = all(abs(e) <= 1.15*s + 1e-5 for e, s in zip(errors, scales))
        self.wait = self.wait + 1 if arrived else 0
        self.stage_age += 1

        cycle_start = 8
        paddle_phase = (self.stage - cycle_start) % 7 if self.stage >= cycle_start else -1
        age_limit = 24 if paddle_phase in (0, 1) else 150
        if (self.small_count > 10 and self.cycles % 2 == 1 and
                0 < self.best_right < (self.small_count + 1) // 2):
            age_limit = 150
        if self.small_count <= 10 and self.stage >= cycle_start and (self.stage - cycle_start) % 6 == 0:
            age_limit = 220
        if self.wait >= settle or self.stage_age > age_limit:
            old_stage = self.stage
            self.stage += 1
            self.wait = 0
            self.stage_age = 0
            cycle_len = 6 if self.small_count <= 10 else 7
            if old_stage >= cycle_start and (old_stage - cycle_start) % cycle_len == cycle_len - 1:
                self.cycles += 1
            # Recompute the floor-level base height because the dynamic hook can
            # slip slightly in the grasp while pouring.
            if self.stage == 6:
                hy = self._get(state, "hook", "y")
                self.low_y = float(np.clip(ry + 0.03 - hy, 0.72, 1.20))
                hook_dx = self._get(state, "hook", "x") - rx
                self.left_x = 0.55 - hook_dx
                desired = (1.60 if self.small_count <= 10 else 1.48) - hook_dx
                self.sweep_x = min(desired, 1.49)
            # Pouring changes the dynamic hook's offset in the fingers.  Refresh
            # the sparse scoop's floor and x alignment before each return descent.
            if (self.small_count <= 10 and self.stage >= 8 and
                    (self.stage - 8) % 6 == 4):
                hook_dx = self._get(state, "hook", "x") - rx
                hook_dy = self._get(state, "hook", "y") - ry
                self.low_y = float(np.clip(0.03 - hook_dy, 0.72, 1.20))
                self.left_x = 0.55 - hook_dx
                self.sweep_x = min(1.60 - hook_dx, 1.49)
            if self.small_count > 10 and self.stage >= 8 and (self.stage - 8) % 7 == 3:
                # A released hook can settle just beyond the wall-side base
                # limit.  Keep the gripper center reachable; its open span still
                # covers the small residual offset.
                self.regrasp_x = float(np.clip(
                    self._get(state, "hook", "x"),
                    0.25, 1.48))
                self.regrasp_y = 0.75
            x, y, theta, joint, gap, settle = self._waypoint()
            errors = (x-rx, y-ry, self._angle_error(theta-rt), joint-rj, gap-rg)

        action = np.asarray([
            np.clip(errors[0], -0.03, 0.03),
            np.clip(errors[1], -0.03, 0.03),
            np.clip(errors[2], -0.098, 0.098),
            np.clip(errors[3], -0.08, 0.08),
            np.clip(errors[4], -0.015, 0.015),
        ], dtype=float)
        if self.small_count <= 10 and self.stage >= cycle_start and (self.stage - cycle_start) % 6 == 0:
            lift_speed = 0.01 if self.cycles == 0 else 0.015
            action[1] = np.clip(errors[1], -lift_speed, lift_speed)
        return np.clip(action, self.low, self.high).astype(
            self.action_dtype, copy=False)
