"""Fast reactive pick-and-place policy for ClutteredStorage2DEnv."""

import math
import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.obs_space = observation_space
        self.low = np.asarray(action_space.low, dtype=np.float32)
        self.high = np.asarray(action_space.high, dtype=np.float32)
        self.robot_type = observation_space.get_type("crv_robot")
        self.shelf_type = observation_space.get_type("shelf")
        self.block_type = observation_space.get_type("target_block")

    @staticmethod
    def _angle(a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _get(self, state, obj, feature):
        return float(state.get(obj, feature))

    def _blocks(self, state):
        return list(state.get_objects(self.block_type))

    def _inside(self, state, block):
        shelf = state.get_objects(self.shelf_type)[0]
        x = self._get(state, block, "x")
        y = self._get(state, block, "y")
        sx = self._get(state, shelf, "x1")
        sy = self._get(state, shelf, "y1")
        sw = self._get(state, shelf, "width1")
        sh = self._get(state, shelf, "height1")
        return sx <= x <= sx + sw and sy <= y <= sy + sh

    def reset(self, state, info):
        self.stage = "choose"
        self.target_name = None
        self.approach_theta = 0.0
        self.goal_base = (0.0, 0.0)
        self.extend_ticks = 0
        self.last_arm = None
        self.stall_ticks = 0
        self.grasp_offset = 0.0
        self.place = (2.5, 2.67)
        self.preplace = (2.5, 2.30)
        self.stage_ticks = 0
        self.attempts = {}
        self.regrasped = False
        # Blocks whose x is behind the world boundary cannot be approached by
        # the circular base.  They are already goals and need not be removed;
        # leave them packed while clearing reachable mouth obstructions.
        self.to_clear = [b.name for b in self._blocks(state) if self._inside(state, b)]
        self.clear_mode = False
        self.priority_name = None
        self.direct_mode = len(self._blocks(state)) >= 5
        self.direct_pick = False
        stored = [b for b in self._blocks(state) if self._inside(state, b)]
        outside = [b for b in self._blocks(state) if not self._inside(state, b)]
        if self.direct_mode:
            for b in outside:
                if abs(self._angle(self._get(state, b, "theta"))) > 2.5:
                    self.attempts[b.name] = 1
        blockers = [q for q in outside for b in stored
                    if abs(self._get(state, q, "x")-self._get(state, b, "x")) < 0.45
                    and self._get(state, q, "y") > 1.60]
        self.blocker_name = blockers[0].name if blockers else None
        self.staging_mode = False

    def _action(self, dx=0.0, dy=0.0, dt=0.0, da=0.0, vac=0.0):
        a = np.asarray([dx, dy, dt, da, vac], dtype=np.float32)
        return np.minimum(self.high, np.maximum(self.low, a)).astype(np.float32)

    def _choose_slot(self, state, exclude):
        shelf = state.get_objects(self.shelf_type)[0]
        # x1 is a left edge (unlike the main rectangle's x convention).
        x1 = self._get(state, shelf, "x1")
        w1 = self._get(state, shelf, "width1")
        y1 = self._get(state, shelf, "y1")
        # Goal coordinates are lower-left bounds. The vertical payload needs
        # only its narrow half-width inside the opening.
        # At insertion the horizontal vacuum bar is 0.14 wide; retain a tiny
        # jamb margin while using the remaining space beside stored blocks.
        # During insertion the base shares the payload x coordinate and must
        # remain at least one radius inside the world boundary.
        lo, hi = max(x1 + 0.071, 0.23), min(x1 + w1 - 0.071, 4.77)
        if hi <= lo:
            return x1 + w1 / 2.0, y1 + 0.10
        occupied = []
        for b in self._blocks(state):
            if b.name == exclude or not self._inside(state, b):
                continue
            theta = self._get(state, b, "theta")
            half_width = 0.5 * (abs(math.cos(theta)) * self._get(state, b, "width") +
                                abs(math.sin(theta)) * self._get(state, b, "height"))
            occupied.append((self._get(state, b, "x"), half_width))
        if not occupied:
            return x1 + w1 / 2.0, y1 + 0.10
        candidates = np.linspace(lo, hi, 25)
        def clearance(q):
            return min((abs(float(q) - z) - h for z, h in occupied), default=99.0)
        feasible = [q for q in candidates if clearance(q) >= 0.073]
        # Pack from the far edge inward.  This preserves one contiguous gap
        # and greatly shortens later transports in wide, high-count shelves.
        x = max(feasible) if feasible else max(candidates, key=clearance)
        return float(x), y1 + 0.10

    def get_action(self, state):
        robot = state.get_objects(self.robot_type)[0]
        rx, ry = self._get(state, robot, "x"), self._get(state, robot, "y")
        rt = self._get(state, robot, "theta")
        arm = self._get(state, robot, "arm_joint")
        blocks = self._blocks(state)

        if self.stage == "choose":
            outside = [b for b in blocks if not self._inside(state, b)]
            clear = [b for b in blocks if b.name in self.to_clear]
            if not outside and not clear:
                return self._action()
            priority = [b for b in outside if b.name == self.priority_name]
            blocker = [b for b in outside if b.name == self.blocker_name]
            self.staging_mode = bool(blocker) and bool(clear)
            self.clear_mode = bool(clear) and not priority and not self.staging_mode
            pool = (priority if priority else
                    (blocker if self.staging_mode else (clear if clear else outside)))
            if priority:
                self.priority_name = None
            if self.clear_mode:
                b = max(pool, key=lambda z: self._get(state, z, "x"))
            else:
                b = min(pool, key=lambda z: (self._get(state, z, "x") - rx) ** 2 +
                                             (self._get(state, z, "y") - ry) ** 2)
            if self.target_name != b.name:
                self.regrasped = False
                self.relocating = False
                self.was_relocated = False
            self.target_name = b.name
            self.direct_routed = False
            bx, by = self._get(state, b, "x"), self._get(state, b, "y")
            self.boundary_clear = self.clear_mode and bx < 0.21
            self.direct_pick = (self.direct_mode and not self.clear_mode and by < 2.0)
            # A radial/end-on pickup can push one member of an overlapping
            # floor cluster into the wall or under another block.  Treat such
            # targets as a retry so the broad-face approach separates them.
            if self.direct_pick:
                crowded = any(
                    q.name != b.name and not self._inside(state, q) and
                    math.hypot(self._get(state, q, "x") - bx,
                               self._get(state, q, "y") - by) < 0.30
                    for q in blocks)
                if crowded:
                    self.direct_pick = False
                    self.attempts[b.name] = max(1, self.attempts.get(b.name, 0))
            if self.clear_mode:
                # A stored horizontal block is only accessible through the
                # lower opening.  A slight tilt lets the vacuum face pass the
                # shelf lip; a mathematically vertical approach catches it.
                th = 3.0 * math.pi / 4.0 if self.boundary_clear else 1.45
            else:
                attempt = self.attempts.get(b.name, 0)
                if self.direct_pick and attempt > 0:
                    self.direct_pick = False
                if self.direct_pick:
                    btheta = self._get(state, b, "theta")
                    qx = bx - .54 * math.cos(btheta)
                    qy = by - .54 * math.sin(btheta)
                    if 0.22 <= qx <= 4.78 and 0.22 <= qy <= 2.3:
                        th = btheta
                    else:
                        self.direct_pick = False
                        th = math.atan2(by-ry, bx-rx)
                elif attempt == 0:
                    th = math.atan2(by - ry, bx - rx)
                else:
                    btheta = self._get(state, b, "theta")
                    options = [self._angle(btheta + math.pi / 2),
                               self._angle(btheta - math.pi / 2)]
                    def cost(q):
                        qx, qy = bx - .54 * math.cos(q), by - .54 * math.sin(q)
                        edge = (max(0.0, .22-qx) + max(0.0, qx-4.78) +
                                max(0.0, .22-qy) + max(0.0, qy-2.3)) * 50.0
                        return edge + math.hypot(qx-rx, qy-ry)
                    th = min(options, key=cost)
            self.approach_theta = th
            self.goal_base = (((bx + 0.48, by - 0.48) if self.boundary_clear
                               else (bx, 2.10)) if self.clear_mode else
                              (bx - 0.54 * math.cos(th), by - 0.54 * math.sin(th)))
            self.direct_offset_pending = self.direct_pick and abs(th) > 2.5
            self.place = self._choose_slot(state, b.name)
            shelf = state.get_objects(self.shelf_type)[0]
            lower = self._get(state, shelf, "y1") - self._get(state, shelf, "height1") / 2.0
            self.shelf_lower = lower
            self.shelf_y1 = self._get(state, shelf, "y1")
            self.shelf_x1 = self._get(state, shelf, "x1")
            self.preplace = (self.place[0], 2.29)
            other_x = [self._get(state, q, "x") for q in blocks if q.name != b.name]
            side_candidates = (0.25, 4.75)
            self.clear_side_x = max(side_candidates,
                                    key=lambda q: min((abs(q-z) for z in other_x),
                                                      default=99.0))
            self.stage = "clear_low" if self.clear_mode else "navigate"
            self.stage_ticks = 0
            self.nav_last = (rx, ry)
            self.nav_stall = 0

        target = state.get_object_from_name(self.target_name)
        bx, by = self._get(state, target, "x"), self._get(state, target, "y")
        bt = self._get(state, target, "theta")

        if self.stage == "clear_low":
            if math.hypot(rx-self.nav_last[0], ry-self.nav_last[1]) < 0.001:
                self.nav_stall += 1
            else:
                self.nav_stall = 0
            self.nav_last = (rx, ry)
            if self.nav_stall >= 3:
                self.stage = ("clear_high_align" if ry > 2.10
                              else "clear_escape_down")
                self.nav_stall = 0
                return self._action(vac=0.0)
            ey = 2.0 - ry
            if abs(ey) < 0.01:
                self.stage = "clear_across"
                return self._action(vac=0.0)
            return self._action(dy=ey, da=0.2-arm, vac=0.0)

        if self.stage == "clear_high_align":
            ex = self.goal_base[0]-rx
            if abs(ex) < 0.01:
                self.stage = "clear_low"
                self.nav_last = (rx, ry)
                self.nav_stall = 0
                return self._action(vac=0.0)
            return self._action(dx=ex, da=0.2-arm, vac=0.0)

        if self.stage == "clear_across":
            ex = self.goal_base[0] - rx
            et = self._angle(self.approach_theta - rt)
            if math.hypot(rx-self.nav_last[0], ry-self.nav_last[1]) < 0.001:
                self.nav_stall += 1
            else:
                self.nav_stall = 0
            self.nav_last = (rx, ry)
            if self.nav_stall >= 4:
                # Move laterally away from the blocking floor object first;
                # descending while it is directly below wedges the base.
                self.stage = "clear_escape_high"
                return self._action(vac=0.0)
            if abs(ex) < 0.01 and abs(et) < 0.02:
                self.stage = "navigate"
                self.nav_last = (rx, ry)
                self.nav_stall = 0
                return self._action(vac=0.0)
            return self._action(dx=ex, dt=et, da=0.2-arm, vac=0.0)

        if self.stage == "clear_escape_down":
            if ry <= 0.46:
                self.stage = "clear_escape_under"
                return self._action(vac=0.0)
            return self._action(dy=0.45-ry, vac=0.0)

        if self.stage == "clear_escape_across":
            if abs(self.clear_side_x-rx) < 0.01:
                self.stage = "clear_escape_down"
                return self._action(vac=0.0)
            return self._action(dx=self.clear_side_x-rx, vac=0.0)

        if self.stage == "clear_escape_under":
            if abs(self.goal_base[0]-rx) < 0.01:
                self.stage = "clear_escape_up_target"
                return self._action(vac=0.0)
            return self._action(dx=self.goal_base[0]-rx, vac=0.0)

        if self.stage == "clear_escape_up_target":
            if abs(2.10-ry) < 0.01:
                self.stage = "navigate"
                self.nav_last = (rx, ry)
                self.nav_stall = 0
                return self._action(vac=0.0)
            return self._action(dy=2.10-ry, vac=0.0)

        if self.stage == "clear_up":
            ey = 2.10 - ry
            if abs(ey) < 0.01:
                self.stage = "navigate"
                self.nav_last = (rx, ry)
                self.nav_stall = 0
                return self._action(vac=0.0)
            return self._action(dy=ey, da=0.2-arm, vac=0.0)

        if self.stage == "navigate":
            if self.direct_offset_pending and self.stage_ticks >= 1:
                offset = -0.08
                self.goal_base = (self.goal_base[0] - offset * math.sin(self.approach_theta),
                                  self.goal_base[1] + offset * math.cos(self.approach_theta))
                self.direct_offset_pending = False
            gx, gy = self.goal_base
            routing = False
            if (self.direct_pick and not self.direct_routed and
                    abs(math.sin(self.approach_theta)) > 0.90):
                ux, uy = math.cos(self.approach_theta), math.sin(self.approach_theta)
                along = (rx-bx)*ux + (ry-by)*uy
                if along > -0.30:
                    tx, ty = -uy, ux
                    side = 1.0 if (rx-bx)*tx + (ry-by)*ty >= 0.0 else -1.0
                    wx = min(4.75, max(0.25, bx-0.65*ux+side*0.65*tx))
                    wy = min(2.25, max(0.25, by-0.65*uy+side*0.65*ty))
                    if max(abs(wx-rx), abs(wy-ry)) >= 0.01:
                        gx, gy = wx, wy
                        routing = True
                    else:
                        self.direct_routed = True
            ex, ey = gx - rx, gy - ry
            et = self._angle(self.approach_theta - rt)
            ea = 0.2 - arm
            self.stage_ticks += 1
            if math.hypot(rx-self.nav_last[0], ry-self.nav_last[1]) < 0.001:
                self.nav_stall += 1
            else:
                self.nav_stall = 0
            self.nav_last = (rx, ry)
            contact_stall = (self.nav_stall >= 2 and abs(et) < 0.012 and
                             math.hypot(bx-rx, by-ry) < 0.85)
            if self.clear_mode and self.nav_stall >= 4 and not contact_stall:
                self.stage = "clear_escape_high"
                return self._action(vac=0.0)
            if not self.clear_mode and self.nav_stall >= 4 and not contact_stall:
                self.stage = "outside_escape"
                self.stage_ticks = 0
                return self._action(vac=0.0)
            if ((max(abs(ex), abs(ey)) < 0.008 and abs(et) < 0.012 and abs(ea) < 0.008)
                    or contact_stall):
                if routing:
                    self.direct_routed = True
                    return self._action(dx=self.goal_base[0]-rx,
                                        dy=self.goal_base[1]-ry, dt=et, da=ea, vac=0.0)
                if self.clear_mode and contact_stall and math.hypot(ex, ey) > 0.05:
                    self.approach_theta = math.atan2(by-ry, bx-rx)
                    self.stage = "clear_realign"
                    return self._action(vac=0.0)
                self.stage = "extend"
                self.extend_ticks = 0
                self.last_arm = arm
                self.stall_ticks = 0
                self.acquire_pose = (bx, by, bt)
                return self._action(da=math.hypot(bx-rx, by-ry)-arm,
                                    vac=0.0 if self.clear_mode and not self.boundary_clear else 1.0)
            return self._action(0.0 if abs(ex) < 0.008 else ex,
                                0.0 if abs(ey) < 0.008 else ey,
                                et, ea, 0.0)

        if self.stage == "clear_realign":
            et = self._angle(self.approach_theta-rt)
            if abs(et) < 0.012:
                self.stage = "extend"
                self.extend_ticks = 0
                self.last_arm = arm
                self.stall_ticks = 0
                self.acquire_pose = (bx, by, bt)
                return self._action(vac=0.0)
            return self._action(dt=et, da=0.2-arm, vac=0.0)

        if self.stage == "outside_escape":
            self.stage_ticks += 1
            if self.stage_ticks <= 6:
                sx = -0.05 if bx >= rx else 0.05
                return self._action(dx=sx, dy=-0.05, da=0.2-arm, vac=0.0)
            self.stage = "navigate"
            self.nav_last = (rx, ry)
            self.nav_stall = 0
            return self._action(vac=0.0)

        if self.stage == "clear_escape_high":
            if ry < 2.21:
                return self._action(dy=2.22-ry, vac=0.0)
            if abs(self.goal_base[0]-rx) >= 0.01:
                return self._action(dx=self.goal_base[0]-rx, vac=0.0)
            self.stage = "navigate"
            self.nav_last = (rx, ry)
            self.nav_stall = 0
            return self._action(vac=0.0)

        if self.stage == "extend":
            self.extend_ticks += 1
            moved = (math.hypot(bx - self.acquire_pose[0], by - self.acquire_pose[1]) > 0.004 or
                     abs(self._angle(bt - self.acquire_pose[2])) > 0.01)
            if moved and not self.clear_mode:
                self.grasp_offset = self._angle(bt - rt)
                if self.direct_pick:
                    self.regrasped = True
                self.stage = "retract"
                self.retract_last = arm
                self.retract_stall = 0
                self.retract_escape = 0
                return self._action(da=0.2-arm, vac=1.0)
            if self.last_arm is not None and abs(arm - self.last_arm) < 0.002:
                self.stall_ticks += 1
            else:
                self.stall_ticks = 0
            self.last_arm = arm
            if self.stall_ticks >= 2 or self.extend_ticks >= 10:
                self.stage = "grip"
                return self._action(vac=1.0)
            center_distance = math.hypot(bx - rx, by - ry)
            return self._action(da=center_distance - arm,
                                vac=0.0 if self.clear_mode and not self.boundary_clear else 1.0)

        if self.stage == "grip":
            self.acquire_pose = (bx, by, bt)
            self.stage = "acquire"
            self.stage_ticks = 0
            self.clear_acq_phase = 0
            return self._action(vac=1.0)

        if self.stage == "acquire":
            self.stage_ticks += 1
            moved = (math.hypot(bx - self.acquire_pose[0], by - self.acquire_pose[1]) > 0.004 or
                     abs(self._angle(bt - self.acquire_pose[2])) > 0.01)
            if moved:
                self.grasp_offset = self._angle(bt - rt)
                if self.direct_pick and not self.clear_mode:
                    self.regrasped = True
                self.stage = ("boundary_shift" if self.boundary_clear else
                              ("clear_pull" if self.clear_mode else "retract"))
                self.retract_last = arm
                self.retract_stall = 0
                self.retract_escape = 0
                self.clear_last_y = by
                self.clear_pull_stall = 0
                if self.clear_mode or self.boundary_clear:
                    return self._action(vac=1.0)
                return self._action(da=0.2-arm, vac=1.0)
            if self.clear_mode:
                # Seat the energized gripper by translating the base.  The
                # block's first motion is our only attachment signal.
                if self.boundary_clear:
                    return self._action(dy=0.02, vac=1.0)
                return self._action(dy=0.02, vac=1.0)
            if self.direct_pick and self.stage_ticks > 20:
                self.attempts[self.target_name] = self.attempts.get(self.target_name, 0) + 1
                d = max(1e-6, math.hypot(rx-bx, ry-by))
                self.retreat_dir = ((rx-bx)/d, (ry-by)/d)
                self.stage = "failed_retreat"
                self.stage_ticks = 0
                return self._action(da=0.2-arm, vac=0.0)
            if self.stage_ticks > 96:
                self.attempts[self.target_name] = self.attempts.get(self.target_name, 0) + 1
                d = max(1e-6, math.hypot(rx - bx, ry - by))
                self.retreat_dir = ((rx - bx) / d, (ry - by) / d)
                self.stage = "failed_retreat"
                self.stage_ticks = 0
                return self._action(da=0.2 - arm, vac=0.0)
            if self.stage_ticks > 32:
                phase = ((self.stage_ticks - 33) // 4) % 4
                dt, da = ((0.04, 0.0), (-0.04, 0.0),
                          (0.0, -0.02), (0.0, 0.02))[phase]
                vac = 0.0 if self.stage_ticks % 24 == 0 else 1.0
                return self._action(dt=dt, da=da, vac=vac)
            # Slide the energized contact in each world direction. Vacuum
            # acquisition happens on overlap, not on first collision.
            phase = ((self.stage_ticks - 1) // 4) % 4
            ux, uy = math.cos(rt), math.sin(rt)
            tx, ty = -uy, ux
            dx, dy = ((0.025*ux, 0.025*uy), (-0.025*ux, -0.025*uy),
                      (0.025*tx, 0.025*ty), (-0.025*tx, -0.025*ty))[phase]
            return self._action(dx, dy, 0.0, 0.0, 1.0)

        if self.stage == "failed_retreat":
            self.stage_ticks += 1
            if self.stage_ticks <= 6:
                return self._action(dx=0.05 * self.retreat_dir[0],
                                    dy=0.05 * self.retreat_dir[1],
                                    da=0.2 - arm, vac=0.0)
            self.stage = "choose"
            return self._action(vac=0.0)

        if self.stage == "retract":
            if arm > 0.23:
                if abs(arm-self.retract_last) < 0.002:
                    self.retract_stall += 1
                else:
                    self.retract_stall = 0
                self.retract_last = arm
                if self.retract_stall >= 3 and self.retract_escape < 10:
                    others = [q for q in blocks if q.name != self.target_name and
                              not self._inside(state, q)]
                    if others:
                        near = min(others, key=lambda q:
                                   (self._get(state, q, "x")-rx)**2 +
                                   (self._get(state, q, "y")-ry)**2)
                        vx = rx-self._get(state, near, "x")
                        vy = ry-self._get(state, near, "y")
                        d = max(1e-6, math.hypot(vx, vy))
                        self.retract_escape += 1
                        return self._action(dx=.05*vx/d, dy=.05*vy/d, vac=1.0)
                if self.retract_escape >= 10:
                    self.retract_stall = 0
                    self.retract_escape = 0
                return self._action(da=0.2 - arm, vac=1.0)
            if self.clear_mode:
                self.stage = "clear_lift"
                return self._action(vac=1.0)
            self.stage = "orient"
            self.orient_last_rt = rt
            self.orient_stall = 0
            self.orient_escape = 0
            return self._action(vac=1.0)

        if self.stage == "clear_pull":
            # Once seated, pull the whole base straight back.  Joint
            # retraction is blocked by the lip, but base translation extracts
            # the block through the opening.
            if abs(by-self.clear_last_y) < 0.001:
                self.clear_pull_stall += 1
            else:
                self.clear_pull_stall = 0
            self.clear_last_y = by
            if by > 2.25 and self.clear_pull_stall < 4:
                return self._action(dy=-0.03, vac=1.0)
            self.to_clear = [n for n in self.to_clear if n != self.target_name]
            self.priority_name = self.target_name
            self.stage = "clear_detach"
            return self._action(vac=0.0)

        if self.stage == "boundary_shift":
            if by > 2.55:
                return self._action(dy=-0.03, vac=1.0)
            if bx < self.shelf_x1 + 0.17:
                return self._action(dt=-0.03, vac=1.0)
            self.stage = "clear_pull"
            self.clear_last_y = by
            self.clear_pull_stall = 0
            return self._action(dy=-0.03, vac=1.0)

        if self.stage == "clear_detach":
            # Give the joint a stationary release step before withdrawing;
            # retracting simultaneously can drag the payload back down.
            self.stage = "clear_release"
            return self._action(vac=0.0)

        if self.stage == "clear_release":
            if arm > 0.21:
                return self._action(da=0.2 - arm, vac=0.0)
            self.stage = "choose"
            self.target_name = None
            return self._action(vac=0.0)

        if self.stage == "orient":
            stored_count = sum(self._inside(state, q) for q in blocks)
            if (not self.regrasped and stored_count >= 4 and
                    abs(self.grasp_offset) < 0.20):
                self.regrasped = True
            # Finish with the base below the payload.  Either vertical block
            # orientation fits; the arm must point upward for shelf approach.
            desired_rt = (math.pi / 2.0 if self.regrasped
                          else self._angle(-self.grasp_offset))
            et = self._angle(desired_rt - rt)
            if abs(et) >= 0.025:
                if abs(self._angle(rt-self.orient_last_rt)) < 0.002:
                    self.orient_stall += 1
                else:
                    self.orient_stall = 0
                self.orient_last_rt = rt
                if self.orient_stall >= 3 and self.orient_escape < 10:
                    others = [q for q in blocks if q.name != self.target_name and
                              not self._inside(state, q)]
                    if others:
                        near = min(others, key=lambda q:
                                   (self._get(state, q, "x")-rx)**2 +
                                   (self._get(state, q, "y")-ry)**2)
                        vx, vy = rx-self._get(state, near, "x"), ry-self._get(state, near, "y")
                        d = max(1e-6, math.hypot(vx, vy))
                        self.orient_escape += 1
                        return self._action(dx=.05*vx/d, dy=.05*vy/d, vac=1.0)
                dx = self.preplace[0] - bx if self.direct_pick else 0.0
                dy = min(0.0, 1.85 - by) if not self.regrasped else 0.0
                return self._action(dx=dx, dy=dy, dt=et, vac=1.0)
            if not self.regrasped:
                # Make room for the end-on regrasp before releasing a block
                # extracted next to a world boundary.
                if bx < 0.80:
                    return self._action(dx=0.80 - bx, vac=1.0)
                if bx > 4.20:
                    return self._action(dx=4.20 - bx, vac=1.0)
                low_others = [(self._get(state, q, "x"), self._get(state, q, "y"))
                              for q in blocks
                              if q.name != self.target_name and not self._inside(state, q)]
                side_sign = 1.0 if rx >= bx else -1.0
                trial_x = bx + 0.50 * side_sign
                lo, hi = sorted((rx, trial_x))
                flipped = any(math.hypot(trial_x-x, ry-y) < 0.28 or
                              (self.direct_mode and lo-0.05 <= x <= hi+0.05 and
                               abs(y-ry) < 0.38)
                              for x, y in low_others)
                if flipped:
                    side_sign = -side_sign
                left_goal = bx-0.54
                path_lo, path_hi = sorted((rx, left_goal))
                corridor_blockers = [(x, y) for x, y in low_others
                                     if path_lo-0.05 <= x <= path_hi+0.05 and
                                     abs(y-(by-0.50)) < 0.38]
                would_grasp_right = ((flipped and side_sign > 0) or bx-0.54 < 0.22 or
                                     (self.direct_mode and bool(corridor_blockers)))
                if would_grasp_right and not self.relocating:
                    self.relocation_x = max([2.45] + [x+0.95 for x, y in corridor_blockers])
                    self.relocation_x = min(4.0, self.relocation_x)
                if self.relocating or would_grasp_right:
                    self.relocating = True
                    if abs(self.relocation_x-bx) >= 0.01:
                        return self._action(dx=self.relocation_x-bx, vac=1.0)
                    self.relocating = False
                    self.was_relocated = True
                self.stage = "drop_for_regrasp"
                return self._action(vac=0.0)
            self.stage = "lower_transport"
            ey = 2.25-by
            ex = self.preplace[0]-bx if by > 1.15 else 0.0
            return self._action(dx=ex, dy=ey, vac=1.0)

        if self.stage == "drop_for_regrasp":
            # Regrasp a horizontal block end-on, then rotate it vertically for
            # passage through the shelf mouth.
            self.regrasped = True
            low_others = [(self._get(state, q, "x"), self._get(state, q, "y"))
                          for q in blocks
                          if q.name != self.target_name and not self._inside(state, q)]
            def side_score(sign):
                gx = bx + sign * 0.54
                boundary = min(gx - 0.22, 4.78 - gx)
                clearance = min((abs(gx-x) - 0.30 for x, y in low_others if y < 2.4),
                                default=9.0)
                return min(boundary, clearance)
            side_sign = 1.0 if (self.was_relocated or rx >= bx) else -1.0
            trial_x = bx + 0.50 * side_sign
            lo, hi = sorted((rx, trial_x))
            flipped_side = any(math.hypot(trial_x-x, ry-y) < 0.28 or
                               (self.direct_mode and lo-0.05 <= x <= hi+0.05 and
                                abs(y-ry) < 0.38)
                               for x, y in low_others)
            if flipped_side:
                side_sign = -side_sign
            self.flipped_regrasp = False
            # Always grasp the left end when possible: it yields the required
            # +pi/2 payload orientation with the base below the block.  The
            # side waypoint may still detour right around clutter.
            if (flipped_side and side_sign > 0) or bx - 0.54 < 0.22:
                self.approach_theta = math.pi
                self.goal_base = (bx + 0.54, by)
            else:
                self.approach_theta = 0.0
                self.goal_base = (bx - (0.60 if flipped_side else 0.54), by)
            safe_y = by - 0.50
            if flipped_side:
                safe_y = by - 0.85
            self.regrasp_safe = (self.goal_base[0], max(0.24, safe_y))
            dd = max(1e-6, math.hypot(rx - bx, ry - by))
            self.retreat_dir = ((rx - bx) / dd, (ry - by) / dd)
            self.regrasp_side_x = bx + (0.60 if flipped_side else 0.50) * side_sign
            self.route_side_first = True
            self.regrasp_switched = False
            self.reposition_last_x = None
            self.reposition_stall = 0
            # The preceding orient action already supplied the required
            # stationary vacuum-off detach step, so begin retreat immediately.
            self.stage = "reposition_side"
            return self._action(dx=0.05 * self.retreat_dir[0],
                                dy=0.05 * self.retreat_dir[1], vac=0.0)

        if self.stage == "retreat_regrasp":
            self.stage_ticks += 1
            if self.stage_ticks <= 1:
                return self._action(dx=0.05 * self.retreat_dir[0],
                                    dy=0.05 * self.retreat_dir[1], vac=0.0)
            self.stage = "reposition_side"
            return self._action(vac=0.0)

        if self.stage == "reposition_side":
            ex = self.regrasp_side_x - rx
            if abs(ex) < 0.01:
                self.stage = "reposition_low"
                self.low_last_y = ry
                self.low_stall = 0
                return self._action(dy=self.regrasp_safe[1] - ry,
                                    da=0.2-arm, vac=0.0)
            return self._action(dx=ex, da=0.2-arm, vac=0.0)

        if self.stage == "reposition_low":
            ey = self.regrasp_safe[1] - ry
            if abs(ry-self.low_last_y) < 0.001 and abs(ey) >= 0.01:
                self.low_stall += 1
            else:
                self.low_stall = 0
            self.low_last_y = ry
            if self.low_stall >= 3:
                away = 1.0 if rx >= bx else -1.0
                self.regrasp_side_x = min(4.75, max(0.25, rx+0.25*away))
                self.regrasp_safe = (self.regrasp_safe[0],
                                     max(0.24, self.regrasp_safe[1]-0.25))
                self.stage = "reposition_side"
                return self._action(dx=self.regrasp_side_x-rx, da=0.2-arm, vac=0.0)
            if abs(ey) < 0.01:
                self.stage = "reposition_regrasp"
                return self._action(dx=self.regrasp_safe[0] - rx,
                                    dt=self._angle(self.approach_theta - rt),
                                    da=0.2-arm, vac=0.0)
            return self._action(dy=ey, da=0.2-arm, vac=0.0)

        if self.stage == "reposition_regrasp":
            ex, ey = self.regrasp_safe[0] - rx, self.regrasp_safe[1] - ry
            et = self._angle(self.approach_theta - rt)
            if abs(ex) > 0.03:
                if self.reposition_last_x is not None and abs(rx-self.reposition_last_x) < 0.001:
                    self.reposition_stall += 1
                else:
                    self.reposition_stall = 0
                self.reposition_last_x = rx
                if self.reposition_stall >= 3 and not self.regrasp_switched:
                    sign = 1.0 if self.goal_base[0] < bx else -1.0
                    self.regrasp_switched = True
                    self.approach_theta = math.pi if sign > 0.0 else 0.0
                    self.goal_base = (bx+sign*0.54, by)
                    self.regrasp_safe = (bx+sign*0.54, max(0.24, by-0.50))
                    self.regrasp_side_x = bx+sign*0.50
                    self.stage = "reposition_side"
                    return self._action(dx=self.regrasp_side_x-rx, vac=0.0)
            else:
                self.reposition_stall = 0
            if max(abs(ex), abs(ey)) < 0.01 and abs(et) < 0.02:
                self.stage = "renavigate"
                return self._action(vac=0.0)
            # Move sideways clear of the released block before crossing its y.
            if abs(ex) >= 0.01:
                return self._action(dx=ex, da=0.2 - arm, vac=0.0)
            if abs(ey) >= 0.01:
                return self._action(dy=ey, da=0.2 - arm, vac=0.0)
            return self._action(dt=et, da=0.2 - arm, vac=0.0)

        if self.stage == "renavigate":
            gx, gy = self.goal_base
            ex, ey = gx - rx, gy - ry
            route_theta = (-math.pi / 2.0 if self.flipped_regrasp and abs(ey) >= 0.01
                           else self.approach_theta)
            et = self._angle(route_theta - rt)
            ea = 0.2 - arm
            if max(abs(ex), abs(ey)) < 0.008 and abs(et) < 0.012 and abs(ea) < 0.008:
                self.stage = "reextend"
                self.last_arm = arm
                self.stall_ticks = 0
                self.acquire_pose = (bx, by, bt)
                return self._action(da=math.hypot(bx-rx, by-ry)-arm, vac=1.0)
            return self._action(0.0 if abs(ex) < 0.008 else ex,
                                0.0 if abs(ey) < 0.008 else ey, et, ea, 0.0)

        if self.stage == "reextend":
            moved = (math.hypot(bx - self.acquire_pose[0], by - self.acquire_pose[1]) > 0.004 or
                     abs(self._angle(bt - self.acquire_pose[2])) > 0.01)
            if moved:
                self.grasp_offset = self._angle(bt - rt)
                self.stage = "retract"
                self.retract_last = arm
                self.retract_stall = 0
                self.retract_escape = 0
                return self._action(vac=1.0)
            if abs(arm - self.last_arm) < 0.002:
                self.stall_ticks += 1
            else:
                self.stall_ticks = 0
            self.last_arm = arm
            if self.stall_ticks >= 2:
                self.acquire_pose = (bx, by, bt)
                self.stage = "acquire"
                self.stage_ticks = 0
                return self._action(vac=1.0)
            return self._action(da=math.hypot(bx - rx, by - ry) - arm, vac=1.0)

        if self.stage == "lower_transport":
            if self.staging_mode:
                self.stage = "stage_blocker_move"
                return self._action(vac=1.0)
            ey = 2.25 - by
            if abs(ey) < 0.01:
                self.stage = "across_transport"
                self.transport_last_x = bx
                self.transport_stall = 0
                return self._action(dx=self.preplace[0]-bx, vac=1.0)
            # Above the floor clutter, translate toward the shelf while
            # raising; independent x/y limits make this much faster.
            ex = self.preplace[0] - bx if by > 1.15 else 0.0
            return self._action(dx=ex, dy=ey, vac=1.0)

        if self.stage == "stage_blocker_move":
            ex, ey = 2.0-bx, 1.15-by
            if max(abs(ex), abs(ey)) < 0.01:
                self.stage = "stage_blocker_detach"
                return self._action(vac=0.0)
            return self._action(dx=ex, dy=ey, vac=1.0)

        if self.stage == "stage_blocker_detach":
            self.stage = "stage_blocker_retreat"
            return self._action(vac=0.0)

        if self.stage == "stage_blocker_retreat":
            if rx < 2.60:
                return self._action(dx=2.60-rx, da=0.2-arm, vac=0.0)
            self.blocker_name = None
            self.staging_mode = False
            self.stage = "choose"
            self.target_name = None
            return self._action(vac=0.0)

        if self.stage == "across_transport":
            if abs(bx-self.transport_last_x) < 0.001:
                self.transport_stall += 1
            else:
                self.transport_stall = 0
            self.transport_last_x = bx
            if self.transport_stall >= 3:
                self.stage = "transport_escape_up"
                return self._action(vac=1.0)
            ex = self.preplace[0] - bx
            if abs(ex) < 0.01:
                self.stage = "preplace"
                return self._action(dy=self.preplace[1]-by, vac=1.0)
            return self._action(dx=ex, vac=1.0)

        if self.stage == "transport_escape_up":
            if ry < 2.21:
                return self._action(dy=2.22-ry, vac=1.0)
            self.stage = "transport_escape_side"
            return self._action(vac=1.0)

        if self.stage == "transport_escape_side":
            ex = self.preplace[0]-bx
            if abs(ex) >= 0.01:
                return self._action(dx=ex, vac=1.0)
            self.stage = "insert"
            self.last_insert_y = by
            self.insert_stall = 0
            self.insert_toggle = 0
            return self._action(vac=1.0)

        if self.stage == "preplace":
            ex, ey = self.preplace[0] - bx, self.preplace[1] - by
            if abs(ex) < 0.008 and abs(ey) < 0.008:
                self.stage = "insert"
                self.last_insert_y = by
                self.insert_stall = 0
                self.insert_toggle = 0
                insert_ey = self.place[1]-by
                s = math.sin(rt)
                lim = 0.08 if self.direct_mode else 0.06
                da = max(-lim, min(lim, insert_ey/max(0.15, s)))
                return self._action(dx=self.place[0]-bx, da=da, vac=1.0)
            return self._action(ex, ey, 0.0, 0.0, 1.0)

        if self.stage == "insert":
            ex, ey = self.place[0] - bx, self.place[1] - by
            if abs(by - self.last_insert_y) < 0.001:
                self.insert_stall += 1
            else:
                self.insert_stall = 0
            self.last_insert_y = by
            done_y = (self.shelf_y1 + 0.015 if self.direct_mode
                      else self.place[1] - 0.025)
            if abs(ex) < 0.012 and by >= done_y:
                self.stage = "release"
                return self._action(vac=0.0)
            # Keep the base below the wall. Extension follows the (generally
            # diagonal) arm, so cancel its horizontal component with base dx.
            s = math.sin(rt)
            c = math.cos(rt)
            da = ey / max(0.15, s)
            lim = 0.08 if self.direct_mode else 0.06
            da = max(-lim, min(lim, da))
            # The arm is vertical at insertion, so extension and the tiny
            # horizontal correction are independent and can run together.
            return self._action(dx=ex, da=da, vac=1.0)

        if self.stage == "release":
            if arm > 0.21:
                return self._action(da=0.2 - arm, vac=0.0)
            self.stage = "choose"
            self.target_name = None
            return self._action(dy=-0.03, vac=0.0)

        return self._action()
