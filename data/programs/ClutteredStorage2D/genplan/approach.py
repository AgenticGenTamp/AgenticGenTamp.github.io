"""Policy for ClutteredStorage2DEnv (variable object count).

Strategy
--------
The shelf's opening (the second rectangle of the `shelf` double-rectangle) is
sized so that `num_init_shelf_blocks` blocks fit side by side *horizontally*,
but the config also guarantees

    shelf_width > (num_init_shelf_blocks + 1) * block_height

i.e. every block fits if inserted *vertically* (long axis along y), because then
each block only consumes `block_height` (0.04) of the opening's width, and the
opening is `shelf_height` (0.375) tall which exceeds the block width (0.28).

So: leave alone the blocks that already satisfy the goal predicate; for each
remaining block, pick a free vertical "slot" x-coordinate inside the opening,
grasp the block, carry it below that slot, orient it vertically, extend the arm
up into the shelf, release, retract.

Everything is closed-loop: each phase reads the current state and terminates on
a geometric condition, with a stuck-detector (the env rejects the whole state
update on collision, so a blocked action is a no-op) that triggers escape moves.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Geometry helpers (mirroring tomsgeoms2d.Rectangle semantics used by the env)
# --------------------------------------------------------------------------- #


def _wrap(angle):
    """Wrap to [-pi, pi]."""
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _rect_vertices(x, y, w, h, theta):
    """Vertices of a Rectangle(x, y, width, height, theta).

    In tomsgeoms2d, (x, y) is the *corner* of the rectangle and theta rotates
    about that corner. Local corners are (0,0), (w,0), (w,h), (0,h).
    """
    c, s = np.cos(theta), np.sin(theta)
    out = []
    for lx, ly in ((0.0, 0.0), (w, 0.0), (w, h), (0.0, h)):
        out.append((x + c * lx - s * ly, y + s * lx + c * ly))
    return out


def _rect_center(x, y, w, h, theta):
    c, s = np.cos(theta), np.sin(theta)
    lx, ly = w / 2.0, h / 2.0
    return (x + c * lx - s * ly, y + s * lx + c * ly)


def _corner_from_center(cx, cy, w, h, theta):
    """Inverse of _rect_center: corner (x, y) given desired center."""
    c, s = np.cos(theta), np.sin(theta)
    lx, ly = w / 2.0, h / 2.0
    return (cx - (c * lx - s * ly), cy - (s * lx + c * ly))


def _point_in_axis_rect(px, py, rx, ry, rw, rh, pad=0.0):
    """Containment test for an axis-aligned rectangle (theta == 0)."""
    return (rx - pad) <= px <= (rx + rw + pad) and (ry - pad) <= py <= (ry + rh + pad)


# --------------------------------------------------------------------------- #
# State accessors
# --------------------------------------------------------------------------- #


def _find_robot(state):
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        if obj.type.name == "crv_robot":
            return obj
    # Fall back: search by feature signature.
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        if "arm_joint" in state.type_features[obj.type]:
            return obj
    return None


def _find_shelf(state):
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        if obj.type.name == "shelf":
            return obj
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        if "width1" in state.type_features[obj.type]:
            return obj
    return None


def _find_blocks(state):
    """All count-defining blocks, in a stable order."""
    blocks = []
    for name in sorted(state.get_object_names()):
        obj = state.get_object_from_name(name)
        if obj.type.name == "target_block":
            blocks.append(obj)
    if not blocks:
        # Fall back on the naming convention.
        for name in sorted(state.get_object_names()):
            if name.startswith("block"):
                blocks.append(state.get_object_from_name(name))
    return blocks


class _Shelf:
    """The shelf opening (second rectangle) plus the blocking bar (first)."""

    def __init__(self, state, shelf):
        self.ox = float(state.get(shelf, "x1"))
        self.oy = float(state.get(shelf, "y1"))
        self.ow = float(state.get(shelf, "width1"))
        self.oh = float(state.get(shelf, "height1"))
        self.bar_y = float(state.get(shelf, "y"))
        self.bar_h = float(state.get(shelf, "height"))

    def contains_rect(self, verts, pad=0.0):
        for px, py in verts:
            if not _point_in_axis_rect(
                px, py, self.ox, self.oy, self.ow, self.oh, pad
            ):
                return False
        return True


def _block_geom(state, b):
    return (
        float(state.get(b, "x")),
        float(state.get(b, "y")),
        float(state.get(b, "width")),
        float(state.get(b, "height")),
        float(state.get(b, "theta")),
    )


def _block_in_shelf(state, b, sh, pad=0.0):
    x, y, w, h, th = _block_geom(state, b)
    return sh.contains_rect(_rect_vertices(x, y, w, h, th), pad)


# --------------------------------------------------------------------------- #
# The approach
# --------------------------------------------------------------------------- #


class GeneratedApproach:
    # Phase constants.
    _RETRACT_FOR_TRAVEL = 0
    _GOTO_GRASP_STANDOFF = 1
    _ALIGN_GRASP = 2
    _EXTEND_GRASP = 3
    _LIFT_AWAY = 4
    _GOTO_SLOT = 5
    _ORIENT_VERTICAL = 6
    _INSERT = 7
    _RELEASE = 8
    _WITHDRAW = 9
    _DONE = 10

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives

        low = np.asarray(action_space.low, dtype=np.float64)
        high = np.asarray(action_space.high, dtype=np.float64)
        self._lo = low
        self._hi = high
        self.max_dx = float(high[0])
        self.max_dy = float(high[1])
        self.max_dth = float(high[2])
        self.max_darm = float(high[3])

        self._rng = np.random.default_rng(0)
        self._reset_internal()

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #

    def _reset_internal(self):
        self.phase = self._RETRACT_FOR_TRAVEL
        self.target_block = None       # object name of the block being moved
        self.slot_x = None             # x of the shelf slot assigned to it
        self.grasp_theta = None        # base theta used at grasp time
        self.phase_steps = 0
        self.total_steps = 0
        self.stuck = 0
        self.escape = 0
        self.escape_dir = np.zeros(2)
        self.prev_pose = None
        self.carry_offset = None       # (dx, dy) block center - gripper tip at grasp
        self._last_action = None

    def reset(self, state, info):
        self._reset_internal()
        self._rng = np.random.default_rng(12345)
        return None

    # ------------------------------------------------------------------ #
    # Small utilities
    # ------------------------------------------------------------------ #

    def _act(self, dx=0.0, dy=0.0, dth=0.0, darm=0.0, vac=0.0):
        a = np.array([dx, dy, dth, darm, vac], dtype=np.float32)
        a = np.clip(a, self._lo, self._hi).astype(np.float32)
        self._last_action = a
        return a

    def _clip_step(self, dx, dy):
        """Scale a desired (dx, dy) into the per-step box, preserving direction."""
        n = max(abs(dx) / self.max_dx if self.max_dx > 0 else 0.0,
                abs(dy) / self.max_dy if self.max_dy > 0 else 0.0)
        if n > 1.0:
            dx /= n
            dy /= n
        return dx, dy

    def _robot_state(self, state, robot):
        return (
            float(state.get(robot, "x")),
            float(state.get(robot, "y")),
            float(state.get(robot, "theta")),
            float(state.get(robot, "arm_joint")),
            float(state.get(robot, "base_radius")),
            float(state.get(robot, "arm_length")),
            float(state.get(robot, "gripper_height")),
            float(state.get(robot, "gripper_width")),
        )

    def _tool_tip(self, rx, ry, rth, arm, gw):
        """Approximate center of the outer face of the gripper."""
        reach = arm + gw / 2.0
        return rx + np.cos(rth) * reach, ry + np.sin(rth) * reach

    def _set_phase(self, phase):
        self.phase = phase
        self.phase_steps = 0
        self.stuck = 0
        self.escape = 0

    # ------------------------------------------------------------------ #
    # Slot planning
    # ------------------------------------------------------------------ #

    def _plan_slot(self, state, sh, blocks, moving_name):
        """Choose an x inside the shelf opening for a vertically-standing block.

        Occupancy is measured from the blocks currently satisfying the goal.
        We greedily take the widest free gap and put the slot at its center,
        biased toward whichever end has more room.
        """
        bw = None
        bh = None
        for b in blocks:
            bw = float(state.get(b, "width"))
            bh = float(state.get(b, "height"))
            break
        if bw is None:
            bw, bh = 0.28, 0.04

        margin = 0.012
        need = bh + 2.0 * margin

        occupied = []
        for b in blocks:
            if b.name == moving_name:
                continue
            if not _block_in_shelf(state, b, sh, pad=1e-3):
                continue
            x, y, w, h, th = _block_geom(state, b)
            vs = _rect_vertices(x, y, w, h, th)
            xs = [p[0] for p in vs]
            occupied.append((min(xs), max(xs)))
        occupied.sort()

        # Merge overlapping intervals.
        merged = []
        for lo, hi in occupied:
            if merged and lo <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])

        lo_bound = sh.ox + margin + bh / 2.0
        hi_bound = sh.ox + sh.ow - margin - bh / 2.0
        if hi_bound < lo_bound:
            return 0.5 * (sh.ox + sh.ox + sh.ow)

        # Free gaps between occupied intervals.
        gaps = []
        cursor = sh.ox
        for lo, hi in merged:
            if lo - cursor >= need:
                gaps.append((cursor, lo))
            cursor = max(cursor, hi)
        if (sh.ox + sh.ow) - cursor >= need:
            gaps.append((cursor, sh.ox + sh.ow))

        if gaps:
            gaps.sort(key=lambda g: (g[1] - g[0]), reverse=True)
            g0, g1 = gaps[0]
            # Hug the edge of the widest gap that is farther from the clutter,
            # leaving contiguous room for later blocks.
            if (g1 - g0) > 3.0 * need:
                cand = g0 + bh / 2.0 + margin
            else:
                cand = 0.5 * (g0 + g1)
            return float(np.clip(cand, lo_bound, hi_bound))

        # No clean gap: fall back to sweeping across the opening.
        frac = (self.total_steps // 97) % 5
        return float(lo_bound + (hi_bound - lo_bound) * (frac / 4.0))

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def get_action(self, state):
        self.total_steps += 1
        self.phase_steps += 1

        robot = _find_robot(state)
        shelf_obj = _find_shelf(state)
        blocks = _find_blocks(state)

        if robot is None or shelf_obj is None or not blocks:
            return self._act()

        sh = _Shelf(state, shelf_obj)
        rx, ry, rth, arm, brad, alen, gh, gw = self._robot_state(state, robot)

        # --- stuck detection: a rejected action leaves the pose unchanged ---
        pose = (round(rx, 6), round(ry, 6), round(rth, 6), round(arm, 6))
        if self.prev_pose is not None and pose == self.prev_pose:
            self.stuck += 1
        else:
            self.stuck = 0
        self.prev_pose = pose

        # --- remaining work ---
        outside = [b for b in blocks if not _block_in_shelf(state, b, sh, pad=-1e-4)]
        if not outside:
            # Goal satisfied (or about to be); sit still with vacuum off.
            return self._act(vac=0.0)

        holding = self.phase in (
            self._LIFT_AWAY,
            self._GOTO_SLOT,
            self._ORIENT_VERTICAL,
            self._INSERT,
            self._RELEASE,
        )

        # --- (re)select a target block if needed ---
        names_outside = {b.name for b in outside}
        if self.target_block not in names_outside:
            if holding:
                # We think we're carrying something that no longer needs moving;
                # drop it and restart.
                self._set_phase(self._RELEASE)
            else:
                self.target_block = self._choose_block(state, outside, rx, ry, sh)
                self.slot_x = None
                self._set_phase(self._RETRACT_FOR_TRAVEL)

        tb = None
        if self.target_block is not None:
            try:
                tb = state.get_object_from_name(self.target_block)
            except Exception:
                tb = None
        if tb is None and not holding:
            self.target_block = self._choose_block(state, outside, rx, ry, sh)
            self._set_phase(self._RETRACT_FOR_TRAVEL)
            try:
                tb = state.get_object_from_name(self.target_block)
            except Exception:
                return self._act()

        if self.slot_x is None and self.target_block is not None:
            self.slot_x = self._plan_slot(state, sh, blocks, self.target_block)

        # --- escape maneuver if wedged ---
        if self.escape > 0:
            self.escape -= 1
            dx, dy = self._clip_step(
                self.escape_dir[0] * self.max_dx * 1.0,
                self.escape_dir[1] * self.max_dy * 1.0,
            )
            return self._act(dx, dy, 0.0, 0.0, 1.0 if holding else 0.0)

        if self.stuck >= 6:
            ang = self._rng.uniform(-np.pi, np.pi)
            self.escape_dir = np.array([np.cos(ang), np.sin(ang)])
            self.escape = 6
            self.stuck = 0

        # --- phase timeouts guard against livelock ---
        if self.phase_steps > 420:
            if holding:
                self._set_phase(self._RELEASE)
            else:
                self.slot_x = None
                self.target_block = self._choose_block(state, outside, rx, ry, sh)
                self._set_phase(self._RETRACT_FOR_TRAVEL)

        # ------------------------------------------------------------------ #
        # Phase machine
        # ------------------------------------------------------------------ #

        if self.phase == self._RETRACT_FOR_TRAVEL:
            return self._do_retract_for_travel(arm, brad)

        if self.phase == self._GOTO_GRASP_STANDOFF:
            return self._do_goto_grasp_standoff(state, tb, sh, rx, ry, rth, arm, brad)

        if self.phase == self._ALIGN_GRASP:
            return self._do_align_grasp(state, tb, rx, ry, rth, arm, brad)

        if self.phase == self._EXTEND_GRASP:
            return self._do_extend_grasp(state, tb, rx, ry, rth, arm, brad, alen, gw)

        if self.phase == self._LIFT_AWAY:
            return self._do_lift_away(state, tb, sh, rx, ry, rth, arm, brad)

        if self.phase == self._GOTO_SLOT:
            return self._do_goto_slot(state, tb, sh, rx, ry, rth, arm, brad)

        if self.phase == self._ORIENT_VERTICAL:
            return self._do_orient_vertical(state, tb, sh, rx, ry, rth, arm)

        if self.phase == self._INSERT:
            return self._do_insert(state, tb, sh, rx, ry, rth, arm, alen, gw)

        if self.phase == self._RELEASE:
            return self._do_release(arm, brad)

        if self.phase == self._WITHDRAW:
            return self._do_withdraw(state, rx, ry, arm, brad, sh)

        return self._act()

    # ------------------------------------------------------------------ #
    # Block selection
    # ------------------------------------------------------------------ #

    def _choose_block(self, state, outside, rx, ry, sh):
        """Nearest outside block, preferring ones not tucked under the shelf bar."""
        best = None
        best_cost = None
        for b in outside:
            x, y, w, h, th = _block_geom(state, b)
            cx, cy = _rect_center(x, y, w, h, th)
            d = np.hypot(cx - rx, cy - ry)
            cost = d
            # Slight penalty for blocks very close to the shelf bar (harder to
            # approach from below without collisions).
            if cy > sh.bar_y - 0.35:
                cost += 1.5
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = b
        return best.name if best is not None else None

    # ------------------------------------------------------------------ #
    # Phase implementations
    # ------------------------------------------------------------------ #

    def _do_retract_for_travel(self, arm, brad):
        if arm > brad + 1e-4:
            return self._act(darm=-self.max_darm, vac=0.0)
        self._set_phase(self._GOTO_GRASP_STANDOFF)
        return self._act(vac=0.0)

    def _standoff_for_block(self, state, tb, sh):
        """A base position from which the block is reachable, below the shelf."""
        x, y, w, h, th = _block_geom(state, tb)
        cx, cy = _rect_center(x, y, w, h, th)
        # Approach along the block's short axis (perpendicular to its long axis)
        # from whichever side has more free space -- default from below.
        return cx, cy

    def _do_goto_grasp_standoff(self, state, tb, sh, rx, ry, rth, arm, brad):
        if tb is None:
            self._set_phase(self._RETRACT_FOR_TRAVEL)
            return self._act()
        cx, cy = self._standoff_for_block(state, tb, sh)

        # Stand off at a fixed radius so the extended arm can reach the block.
        stand = brad + 0.22
        # Prefer approaching from below (the shelf is above), unless the block
        # is very low in the world.
        if cy - stand > 0.30:
            tx, ty = cx, cy - stand
        else:
            tx, ty = cx, cy + stand
            ty = min(ty, sh.bar_y - brad - 0.06)

        ex, ey = tx - rx, ty - ry
        dist = np.hypot(ex, ey)
        if dist < 0.035:
            self._set_phase(self._ALIGN_GRASP)
            return self._act(vac=0.0)

        # Keep clear of the shelf bar while traveling.
        if ry > sh.bar_y - brad - 0.02:
            return self._act(0.0, -self.max_dy, 0.0, 0.0, 0.0)

        dx, dy = self._clip_step(ex, ey)
        # Cap vertical motion so we never drift into the bar.
        if ry + dy > sh.bar_y - brad - 0.05:
            dy = min(dy, 0.0)
        return self._act(dx, dy, 0.0, 0.0, 0.0)

    def _desired_grasp_theta(self, state, tb, rx, ry):
        """Point the gripper from the base toward the block center."""
        x, y, w, h, th = _block_geom(state, tb)
        cx, cy = _rect_center(x, y, w, h, th)
        return np.arctan2(cy - ry, cx - rx)

    def _do_align_grasp(self, state, tb, rx, ry, rth, arm, brad):
        if tb is None:
            self._set_phase(self._RETRACT_FOR_TRAVEL)
            return self._act()
        want = self._desired_grasp_theta(state, tb, rx, ry)
        err = _wrap(want - rth)
        if abs(err) < 0.05:
            self.grasp_theta = rth
            self._set_phase(self._EXTEND_GRASP)
            return self._act(vac=0.0)
        dth = float(np.clip(err, -self.max_dth, self.max_dth))
        return self._act(0.0, 0.0, dth, 0.0, 0.0)

    def _do_extend_grasp(self, state, tb, rx, ry, rth, arm, brad, alen, gw):
        """Extend the arm with the vacuum on until the block is attached."""
        if tb is None:
            self._set_phase(self._RETRACT_FOR_TRAVEL)
            return self._act()

        x, y, w, h, th = _block_geom(state, tb)
        cx, cy = _rect_center(x, y, w, h, th)
        tipx, tipy = self._tool_tip(rx, ry, rth, arm, gw)

        # Did we capture it? The block snaps rigidly to the gripper, so a very
        # small tip-to-block distance while the vacuum is on means success.
        gap = np.hypot(cx - tipx, cy - tipy)
        if self._last_action is not None and self._last_action[4] > 0.5:
            if gap < (max(w, h) / 2.0 + 0.06):
                # Record the carry offset for later placement reasoning.
                self.carry_offset = (cx - tipx, cy - tipy)
                self._set_phase(self._LIFT_AWAY)
                return self._act(vac=1.0)

        # Keep re-aiming a little while extending.
        want = np.arctan2(cy - ry, cx - rx)
        err = _wrap(want - rth)
        dth = float(np.clip(err, -self.max_dth, self.max_dth)) * 0.6

        # Distance from base to block center tells us how far to extend.
        need = np.hypot(cx - rx, cy - ry) - gw
        if arm < min(need, alen) - 1e-3 and arm < alen - 1e-6:
            return self._act(0.0, 0.0, dth, self.max_darm, 1.0)

        if arm >= alen - 1e-3 and gap > (max(w, h) / 2.0 + 0.06):
            # Fully extended but still short: creep the base forward.
            dx, dy = self._clip_step(np.cos(rth) * 0.05, np.sin(rth) * 0.05)
            return self._act(dx, dy, dth, 0.0, 1.0)

        return self._act(0.0, 0.0, dth, self.max_darm * 0.5, 1.0)

    def _do_lift_away(self, state, tb, sh, rx, ry, rth, arm, brad):
        """Retract partially so the carried block travels close to the base."""
        target_arm = brad + 0.06
        if arm > target_arm + 1e-3:
            return self._act(0.0, 0.0, 0.0, -self.max_darm, 1.0)
        self._set_phase(self._GOTO_SLOT)
        return self._act(vac=1.0)

    def _do_goto_slot(self, state, tb, sh, rx, ry, rth, arm, brad):
        """Drive to just below the assigned shelf slot, staying under the bar."""
        if self.slot_x is None:
            self.slot_x = self._plan_slot(
                state, sh, _find_blocks(state), self.target_block
            )
        # Stand below the shelf so that extending upward pushes the block in.
        ty = sh.bar_y - brad - 0.10
        tx = self.slot_x

        ex, ey = tx - rx, ty - ry
        if abs(ex) < 0.02 and abs(ey) < 0.03:
            self._set_phase(self._ORIENT_VERTICAL)
            return self._act(vac=1.0)

        # Move horizontally first at a safe height, then rise.
        if abs(ex) > 0.02 and ry > ty - 0.02:
            dx, _ = self._clip_step(ex, 0.0)
            return self._act(dx, 0.0, 0.0, 0.0, 1.0)

        dx, dy = self._clip_step(ex, ey)
        if ry + dy > sh.bar_y - brad - 0.04:
            dy = min(dy, 0.0)
        return self._act(dx, dy, 0.0, 0.0, 1.0)

    def _do_orient_vertical(self, state, tb, sh, rx, ry, rth, arm):
        """Rotate the base so the carried block stands on end (long axis in y).

        The block is rigidly attached, so rotating the base by d rotates the
        block by d. We drive the block's own theta toward +/- pi/2, and we also
        want the arm pointing roughly upward so the insertion stroke goes into
        the shelf.
        """
        if tb is None:
            self._set_phase(self._RELEASE)
            return self._act(vac=1.0)

        x, y, w, h, th = _block_geom(state, tb)

        # Error to nearest vertical orientation for the block.
        err_block = min(
            (_wrap(np.pi / 2.0 - th), _wrap(-np.pi / 2.0 - th)), key=abs
        )
        # Error to get the arm pointing up.
        err_arm = _wrap(np.pi / 2.0 - rth)

        # The block's mounting offset is fixed, so both errors move together;
        # prefer block verticality, then break ties by arm-up.
        if abs(err_block) < 0.06:
            if abs(err_arm) < 0.75:
                self._set_phase(self._INSERT)
                return self._act(vac=1.0)
            # Rotate by a full turn's worth toward arm-up while keeping the
            # block vertical (pi steps keep verticality).
            dth = float(np.clip(err_arm, -self.max_dth, self.max_dth))
            return self._act(0.0, 0.0, dth, 0.0, 1.0)

        dth = float(np.clip(err_block, -self.max_dth, self.max_dth))
        return self._act(0.0, 0.0, dth, 0.0, 1.0)

    def _do_insert(self, state, tb, sh, rx, ry, rth, arm, alen, gw):
        """Extend the arm to push the vertical block up into the shelf opening."""
        if tb is None:
            self._set_phase(self._RELEASE)
            return self._act(vac=1.0)

        x, y, w, h, th = _block_geom(state, tb)
        verts = _rect_vertices(x, y, w, h, th)
        cx, cy = _rect_center(x, y, w, h, th)

        if sh.contains_rect(verts, pad=-2e-3):
            self._set_phase(self._RELEASE)
            return self._act(vac=1.0)

        # Lateral correction: keep the block centered on the slot.
        if self.slot_x is not None:
            ex = self.slot_x - cx
        else:
            ex = 0.0

        # Vertical: we want the block fully above sh.oy.
        target_cy = sh.oy + sh.oh / 2.0
        ey = target_cy - cy

        if arm < alen - 1e-3 and ey > 0.0:
            dx, _ = self._clip_step(np.clip(ex, -0.05, 0.05), 0.0)
            return self._act(dx, 0.0, 0.0, self.max_darm, 1.0)

        # Arm maxed: nudge the base upward/laterally.
        dx, dy = self._clip_step(np.clip(ex, -0.05, 0.05), np.clip(ey, -0.05, 0.05))
        return self._act(dx, dy, 0.0, 0.0, 1.0)

    def _do_release(self, arm, brad):
        # A couple of steps with the vacuum off so the release registers.
        if self.phase_steps <= 2:
            return self._act(vac=0.0)
        self._set_phase(self._WITHDRAW)
        return self._act(vac=0.0)

    def _do_withdraw(self, state, rx, ry, arm, brad, sh):
        """Pull the arm straight back out, then drop down clear of the shelf."""
        if arm > brad + 1e-3:
            return self._act(0.0, 0.0, 0.0, -self.max_darm, 0.0)
        if ry > sh.bar_y - brad - 0.18:
            return self._act(0.0, -self.max_dy, 0.0, 0.0, 0.0)
        # Ready for the next block.
        self.target_block = None
        self.slot_x = None
        self.carry_offset = None
        self._set_phase(self._RETRACT_FOR_TRAVEL)
        return self._act(vac=0.0)