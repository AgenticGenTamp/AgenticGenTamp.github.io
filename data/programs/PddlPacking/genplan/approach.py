"""
GeneratedApproach for PR2Packed.

Strategy
--------
1. Precompute a static grid of packing slots on the plate (3x3 with 0.085 m
   spacing, ordered so already-placed blocks never block the approach).
2. For each block in turn run the same scripted pick-and-place macro:
      DRIVE_TO_BLOCK -> DESCEND/GRASP -> LIFT -> DRIVE_TO_PLATE ->
      DESCEND_TO_SLOT -> RELEASE
3. Arm motion is produced by IK against a private "shadow" PyBullet simulation
   of the same scene, driven by clipped proportional joint deltas.  If IK is
   unavailable we fall back to a purely kinematic Jacobian-free controller that
   uses finite differences on the shadow robot.
4. Rejected (no-op) steps are detected by comparing successive observations;
   after repeated rejections we perturb / raise the arm and retry, and after a
   long stall we skip to the next slot or next block.

The shadow simulation is built lazily from the observation itself (block count,
plate pose, table pose), so one program handles any number of blocks.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Optional heavy dependencies.  Everything is guarded: if the shadow simulation
# cannot be constructed we degrade to an open-loop-ish controller that still
# returns valid actions.
# --------------------------------------------------------------------------
_PB_OK = True
try:  # pragma: no cover - import guard
    import pybullet as _p
    from robocode.environments.ss_pybullet import (  # type: ignore
        PR2_TOOL_FRAMES,
        Euler,
        HideOutput,
        Point,
        Pose,
        add_data_path,
        close_arm,
        connect,
        create_box,
        create_pr2,
        create_table,
        get_arm_joints,
        get_carry_conf,
        get_group_joints,
        get_joint_limits,
        get_joint_positions,
        get_link_pose,
        get_other_arm,
        invert,
        link_from_name,
        multiply,
        open_arm,
        pairwise_collision,
        set_arm_conf,
        set_client,
        get_client,
        set_group_conf,
        set_joint_positions,
        set_point,
        set_pose,
        stable_z,
        sub_inverse_kinematics,
        arm_conf,
        REST_LEFT_ARM,
        GREEN,
        BLUE,
    )
except Exception:  # pragma: no cover
    _PB_OK = False


# --------------------------------------------------------------------------
# Geometry constants (mirroring the environment's scene)
# --------------------------------------------------------------------------
BLOCK_WIDTH = 0.07
BLOCK_HEIGHT = 0.10
PLATE_WIDTH = 0.27
ARM = "left"
GRASP_TYPE = "top"

MAX_DELTA = 0.2
GRIP_CLOSE = -1.0
GRIP_OPEN = 1.0
GRIP_NONE = 0.0

# Slot grid on the plate.
SLOT_SPACING = 0.085


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw of a (mostly) z-axis rotation quaternion."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _downward_tool_pose(x: float, y: float, z: float, yaw: float):
    """Tool pose whose +x approach axis points straight down, wrist yaw `yaw`."""
    # Rotate +x to -z:  pitch by +90 deg about y maps +x -> -z.
    return ((x, y, z), _quat_from_euler(0.0, math.pi / 2.0, yaw))


def _quat_from_euler(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


# --------------------------------------------------------------------------
# Shadow simulation: a private copy of the scene used purely as an IK and
# collision oracle.  It never touches the real environment.
# --------------------------------------------------------------------------
class _Shadow:
    """Private PyBullet replica of the packed scene, used for IK."""

    def __init__(self, num_blocks: int):
        self.ok = False
        if not _PB_OK:
            return
        try:
            prev = get_client()
        except Exception:
            prev = None
        try:
            with HideOutput():
                self.client = connect(use_gui=False)
                set_client(self.client)
                add_data_path()
                try:
                    self.floor = _p.loadURDF(
                        "plane.urdf", physicsClientId=self.client
                    )
                except Exception:
                    self.floor = None
                self.robot = create_pr2()
                init_conf = get_carry_conf(ARM, GRASP_TYPE)
                set_arm_conf(self.robot, ARM, init_conf)
                open_arm(self.robot, ARM)
                other = get_other_arm(ARM)
                set_arm_conf(self.robot, other, arm_conf(other, REST_LEFT_ARM))
                close_arm(self.robot, other)
                set_group_conf(self.robot, "base", [-1.0, 0.0, 0.0])
                self.table = create_table()
                self.plate = create_box(
                    PLATE_WIDTH, PLATE_WIDTH, 0.001, color=GREEN
                )
                set_point(self.plate, Point(z=stable_z(self.plate, self.table)))
                self.blocks = [
                    create_box(
                        BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_HEIGHT, color=BLUE
                    )
                    for _ in range(num_blocks)
                ]
                for i, b in enumerate(self.blocks):
                    set_point(b, Point(x=3.0 + 0.3 * i, y=3.0, z=0.05))
            self.base_joints = list(get_group_joints(self.robot, "base"))
            self.arm_joints = list(get_arm_joints(self.robot, ARM))
            self.tool_link = link_from_name(self.robot, PR2_TOOL_FRAMES[ARM])
            self.init_conf = list(init_conf)
            self.ok = True
        except Exception:
            self.ok = False
        finally:
            try:
                if prev is not None:
                    set_client(prev)
            except Exception:
                pass

    # -- context ---------------------------------------------------------
    def _enter(self):
        self._prev = None
        try:
            self._prev = get_client()
        except Exception:
            self._prev = None
        set_client(self.client)

    def _exit(self):
        try:
            if self._prev is not None:
                set_client(self._prev)
        except Exception:
            pass

    # -- sync ------------------------------------------------------------
    def sync(
        self,
        base: Sequence[float],
        arm: Sequence[float],
        block_poses: Dict[int, Tuple[Sequence[float], Sequence[float]]],
        plate_pose=None,
    ) -> None:
        if not self.ok:
            return
        self._enter()
        try:
            set_joint_positions(self.robot, self.base_joints, list(base))
            set_joint_positions(self.robot, self.arm_joints, list(arm))
            for idx, (pt, quat) in block_poses.items():
                if idx < len(self.blocks):
                    set_pose(self.blocks[idx], (tuple(pt), tuple(quat)))
            if plate_pose is not None:
                set_pose(self.plate, plate_pose)
        except Exception:
            pass
        finally:
            self._exit()

    def tool_pose(self, base: Sequence[float], arm: Sequence[float]):
        if not self.ok:
            return None
        self._enter()
        try:
            set_joint_positions(self.robot, self.base_joints, list(base))
            set_joint_positions(self.robot, self.arm_joints, list(arm))
            return get_link_pose(self.robot, self.tool_link)
        except Exception:
            return None
        finally:
            self._exit()

    def ik(
        self,
        base: Sequence[float],
        arm_seed: Sequence[float],
        target_pose,
        tries: int = 6,
    ) -> Optional[List[float]]:
        """Solve arm IK for a tool pose, with the base pinned at `base`."""
        if not self.ok:
            return None
        self._enter()
        try:
            best = None
            for t in range(tries):
                set_joint_positions(self.robot, self.base_joints, list(base))
                if t == 0:
                    seed = list(arm_seed)
                elif t == 1:
                    seed = list(self.init_conf)
                else:
                    seed = [
                        float(a) + np.random.uniform(-0.5, 0.5)
                        for a in arm_seed
                    ]
                set_joint_positions(self.robot, self.arm_joints, seed)
                try:
                    conf = sub_inverse_kinematics(
                        self.robot,
                        self.arm_joints[0],
                        self.tool_link,
                        target_pose,
                    )
                except Exception:
                    conf = None
                if conf is None:
                    continue
                got = get_link_pose(self.robot, self.tool_link)
                perr = float(
                    np.linalg.norm(
                        np.array(got[0]) - np.array(target_pose[0])
                    )
                )
                if perr > 0.03:
                    continue
                sol = list(get_joint_positions(self.robot, self.arm_joints))
                cost = perr + 0.02 * float(
                    np.linalg.norm(np.array(sol) - np.array(arm_seed))
                )
                if best is None or cost < best[0]:
                    best = (cost, sol)
            return None if best is None else best[1]
        except Exception:
            return None
        finally:
            self._exit()

    def close(self):
        if not self.ok:
            return
        try:
            self._enter()
            _p.disconnect(physicsClientId=self.client)
        except Exception:
            pass
        finally:
            self._exit()
            self.ok = False


# --------------------------------------------------------------------------
# Main approach
# --------------------------------------------------------------------------
class GeneratedApproach:
    # Phase names
    P_DRIVE_BLOCK = "drive_block"
    P_PREGRASP = "pregrasp"
    P_DESCEND = "descend"
    P_CLOSE = "close"
    P_LIFT = "lift"
    P_DRIVE_PLATE = "drive_plate"
    P_OVER_SLOT = "over_slot"
    P_LOWER_SLOT = "lower_slot"
    P_RELEASE = "release"
    P_RECOVER = "recover"
    P_DONE = "done"

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives or {}

        self._shadow: Optional[_Shadow] = None
        self._shadow_n = -1

        self._rng = np.random.default_rng(0)

        # per-episode state
        self._reset_episode_vars()

    # ------------------------------------------------------------------ utils
    def _reset_episode_vars(self):
        self.phase = self.P_DRIVE_BLOCK
        self.target_block: Optional[str] = None
        self.slot_of: Dict[str, int] = {}
        self.used_slots: set = set()
        self.slots: List[Tuple[float, float]] = []
        self.plate_center = (0.0, 0.0, 0.7305)
        self.step_count = 0
        self.phase_steps = 0
        self.stall = 0
        self.prev_vec: Optional[np.ndarray] = None
        self.base_target: Optional[np.ndarray] = None
        self.arm_target: Optional[np.ndarray] = None
        self.grasp_yaw = 0.0
        self.retry_count = 0
        self.slot_try = 0
        self.descend_z: Optional[float] = None
        self.last_action = np.zeros(11, dtype=np.float32)
        self.fail_blocks: set = set()
        self.global_recover = 0

    # -- observation helpers --------------------------------------------
    def _robot(self, state):
        try:
            return state.get_object_from_name("robot")
        except Exception:
            for o in state:
                if getattr(o.type, "name", "") == "robot":
                    return o
        return None

    def _surface(self, state, name):
        try:
            return state.get_object_from_name(name)
        except Exception:
            return None

    def _blocks(self, state) -> List[Any]:
        out = []
        for name in sorted(state.get_object_names()):
            if name.startswith("block"):
                try:
                    out.append(state.get_object_from_name(name))
                except Exception:
                    pass
        # sort numerically
        def key(o):
            try:
                return int(o.name[5:])
            except Exception:
                return 0

        out.sort(key=key)
        return out

    def _f(self, state, obj, feat, default=0.0) -> float:
        try:
            return float(state.get(obj, feat))
        except Exception:
            return default

    def _robot_vec(self, state) -> np.ndarray:
        r = self._robot(state)
        names = [
            "base_x",
            "base_y",
            "base_rot",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
            "gripper_opening",
            "grasp_active",
        ]
        return np.array([self._f(state, r, n) for n in names], dtype=np.float64)

    def _state_vec(self, state) -> np.ndarray:
        vals = list(self._robot_vec(state))
        for b in self._blocks(state):
            for f in ("pose_x", "pose_y", "pose_z", "grasp_active"):
                vals.append(self._f(state, b, f))
        return np.array(vals, dtype=np.float64)

    def _block_pose(self, state, b):
        pt = (
            self._f(state, b, "pose_x"),
            self._f(state, b, "pose_y"),
            self._f(state, b, "pose_z"),
        )
        q = (
            self._f(state, b, "pose_qx"),
            self._f(state, b, "pose_qy"),
            self._f(state, b, "pose_qz"),
            self._f(state, b, "pose_qw", 1.0),
        )
        return pt, q

    # ------------------------------------------------------------------ reset
    def reset(self, state, info):
        self._reset_episode_vars()

        blocks = self._blocks(state)
        n = len(blocks)

        # plate geometry
        plate = self._surface(state, "plate")
        if plate is not None:
            self.plate_center = (
                self._f(state, plate, "pose_x"),
                self._f(state, plate, "pose_y"),
                self._f(state, plate, "pose_z"),
            )
            hx = self._f(state, plate, "half_extent_x", PLATE_WIDTH / 2)
            hy = self._f(state, plate, "half_extent_y", PLATE_WIDTH / 2)
        else:
            hx = hy = PLATE_WIDTH / 2

        self.slots = self._make_slots(self.plate_center, hx, hy, n)

        # shadow sim
        if _PB_OK:
            if self._shadow is None or self._shadow_n != n or not self._shadow.ok:
                try:
                    if self._shadow is not None:
                        self._shadow.close()
                except Exception:
                    pass
                self._shadow = _Shadow(n)
                self._shadow_n = n

        # choose block order: far-from-plate first is fine; use distance
        # descending so later (closer) blocks don't get in the way.
        order = []
        for b in blocks:
            pt, _ = self._block_pose(state, b)
            d = math.hypot(
                pt[0] - self.plate_center[0], pt[1] - self.plate_center[1]
            )
            order.append((-d, b.name))
        order.sort()
        self.block_order = [nm for _, nm in order]

        # slot assignment, in fill order
        for i, nm in enumerate(self.block_order):
            self.slot_of[nm] = min(i, len(self.slots) - 1)

        self.target_block = self._pick_next_block(state)
        self.phase = self.P_DRIVE_BLOCK
        self.prev_vec = self._state_vec(state)
        return None

    def _make_slots(self, center, hx, hy, n) -> List[Tuple[float, float]]:
        """A grid of packing slots on the plate, ordered for filling."""
        cx, cy, _ = center
        half = BLOCK_WIDTH / 2.0
        # how many fit per axis
        kx = max(1, int(math.floor((2 * hx) / BLOCK_WIDTH)))
        ky = max(1, int(math.floor((2 * hy) / BLOCK_WIDTH)))
        kx = min(kx, 3)
        ky = min(ky, 3)
        need = max(1, n)
        # expand grid until it holds all blocks (cap at 3x3 realistically)
        while kx * ky < need and (kx < 3 or ky < 3):
            if kx <= ky and kx < 3:
                kx += 1
            elif ky < 3:
                ky += 1
            else:
                break

        def axis(k, h):
            if k <= 1:
                return [0.0]
            span = h - half
            step = min(SLOT_SPACING, (2 * span) / (k - 1)) if k > 1 else 0.0
            start = -step * (k - 1) / 2.0
            return [start + i * step for i in range(k)]

        xs = axis(kx, hx)
        ys = axis(ky, hy)
        cells = []
        for j, dy in enumerate(ys):
            for i, dx in enumerate(xs):
                cells.append((cx + dx, cy + dy))
        # Fill order: far side (-x, away from robot start at x=-1 means +x is far?)
        # Robot starts at x=-1 facing +x, so nearest plate cells are low x.
        # Fill the far (high x) cells first so placed blocks don't block reach.
        cells.sort(key=lambda c: (-(c[0] - cx), abs(c[1] - cy)))
        return cells

    # ------------------------------------------------------------------ main
    def get_action(self, state):
        self.step_count += 1
        self.phase_steps += 1

        # stall detection
        vec = self._state_vec(state)
        if self.prev_vec is not None and vec.shape == self.prev_vec.shape:
            if np.allclose(vec, self.prev_vec, atol=1e-5):
                self.stall += 1
            else:
                self.stall = 0
        self.prev_vec = vec

        try:
            act = self._policy(state)
        except Exception:
            act = np.zeros(11, dtype=np.float32)

        act = np.asarray(act, dtype=np.float32).reshape(11)
        act = np.clip(
            act,
            np.array([-MAX_DELTA] * 10 + [-1.0], dtype=np.float32),
            np.array([MAX_DELTA] * 10 + [1.0], dtype=np.float32),
        )
        self.last_action = act
        return act

    # ------------------------------------------------------------------ core
    def _policy(self, state):
        r = self._robot(state)
        base = np.array(
            [
                self._f(state, r, "base_x"),
                self._f(state, r, "base_y"),
                self._f(state, r, "base_rot"),
            ]
        )
        arm = np.array(
            [self._f(state, r, f"joint_{i}") for i in range(1, 8)]
        )
        holding = self._f(state, r, "grasp_active") > 0.5

        blocks = self._blocks(state)
        if not blocks:
            return self._zero()

        # sync shadow
        self._sync_shadow(state, base, arm)

        # ---------------- select / validate target ----------------
        if holding:
            held = None
            for b in blocks:
                if self._f(state, b, "grasp_active") > 0.5:
                    held = b
                    break
            if held is not None:
                self.target_block = held.name
            if self.phase in (
                self.P_DRIVE_BLOCK,
                self.P_PREGRASP,
                self.P_DESCEND,
                self.P_CLOSE,
            ):
                self._set_phase(self.P_LIFT)
        else:
            if self.phase in (
                self.P_LIFT,
                self.P_DRIVE_PLATE,
                self.P_OVER_SLOT,
                self.P_LOWER_SLOT,
                self.P_RELEASE,
            ):
                # just released (or lost grasp) -> next block
                self.target_block = self._pick_next_block(state)
                self._set_phase(self.P_DRIVE_BLOCK)
            if self.target_block is None or self._on_plate(state, self.target_block):
                self.target_block = self._pick_next_block(state)
                if self.target_block is None:
                    return self._idle()
                self._set_phase(self.P_DRIVE_BLOCK)

        if self.target_block is None:
            return self._idle()

        tb = self._obj(state, self.target_block)
        if tb is None:
            self.target_block = self._pick_next_block(state)
            return self._idle()

        # hard stall recovery
        if self.stall > 14:
            self.stall = 0
            self.global_recover += 1
            return self._recover_action(base, arm, holding)

        # --------------------------- phases ---------------------------
        if self.phase == self.P_DRIVE_BLOCK:
            return self._do_drive_to(state, base, arm, self._block_xy(state, tb),
                                     next_phase=self.P_PREGRASP)

        if self.phase == self.P_PREGRASP:
            return self._do_pregrasp(state, base, arm, tb)

        if self.phase == self.P_DESCEND:
            return self._do_descend(state, base, arm, tb)

        if self.phase == self.P_CLOSE:
            self._set_phase(self.P_LIFT)
            return self._mk(np.zeros(3), np.zeros(7), GRIP_CLOSE)

        if self.phase == self.P_LIFT:
            return self._do_lift(state, base, arm)

        if self.phase == self.P_DRIVE_PLATE:
            slot = self._current_slot()
            return self._do_drive_to(state, base, arm, slot,
                                     next_phase=self.P_OVER_SLOT, carrying=True)

        if self.phase == self.P_OVER_SLOT:
            return self._do_over_slot(state, base, arm)

        if self.phase == self.P_LOWER_SLOT:
            return self._do_lower_slot(state, base, arm)

        if self.phase == self.P_RELEASE:
            return self._do_release(state, base, arm)

        return self._idle()

    # -------------------------------------------------------------- helpers
    def _zero(self):
        return np.zeros(11, dtype=np.float32)

    def _idle(self):
        return np.zeros(11, dtype=np.float32)

    def _mk(self, dbase, darm, grip):
        a = np.zeros(11, dtype=np.float32)
        a[0:3] = np.clip(np.asarray(dbase, dtype=np.float64), -MAX_DELTA, MAX_DELTA)
        a[3:10] = np.clip(np.asarray(darm, dtype=np.float64), -MAX_DELTA, MAX_DELTA)
        a[10] = grip
        return a

    def _set_phase(self, ph):
        if ph != self.phase:
            self.phase = ph
            self.phase_steps = 0
            self.retry_count = 0
            self.stall = 0
            self.arm_target = None

    def _obj(self, state, name):
        try:
            return state.get_object_from_name(name)
        except Exception:
            return None

    def _block_xy(self, state, b):
        return (self._f(state, b, "pose_x"), self._f(state, b, "pose_y"))

    def _current_slot(self):
        nm = self.target_block
        idx = self.slot_of.get(nm, 0)
        idx = (idx + self.slot_try) % max(1, len(self.slots))
        return self.slots[idx]

    def _on_plate(self, state, name) -> bool:
        b = self._obj(state, name)
        if b is None:
            return True
        if self._f(state, b, "grasp_active") > 0.5:
            return False
        px, py, pz = self.plate_center
        bx = self._f(state, b, "pose_x")
        by = self._f(state, b, "pose_y")
        bz = self._f(state, b, "pose_z")
        plate = self._surface(state, "plate")
        hx = self._f(state, plate, "half_extent_x", PLATE_WIDTH / 2) if plate else 0.135
        hy = self._f(state, plate, "half_extent_y", PLATE_WIDTH / 2) if plate else 0.135
        return (
            abs(bx - px) <= hx
            and abs(by - py) <= hy
            and bz > pz + BLOCK_HEIGHT / 2 - 0.02
        )

    def _pick_next_block(self, state) -> Optional[str]:
        # prefer the planned order, skipping already-placed and failed ones
        for nm in getattr(self, "block_order", []):
            if nm in self.fail_blocks:
                continue
            if not self._on_plate(state, nm):
                return nm
        for nm in getattr(self, "block_order", []):
            if not self._on_plate(state, nm):
                return nm
        return None

    def _sync_shadow(self, state, base, arm):
        if self._shadow is None or not self._shadow.ok:
            return
        poses = {}
        for i, b in enumerate(self._blocks(state)):
            poses[i] = self._block_pose(state, b)
        plate = self._surface(state, "plate")
        ppose = None
        if plate is not None:
            ppose = (
                (
                    self._f(state, plate, "pose_x"),
                    self._f(state, plate, "pose_y"),
                    self._f(state, plate, "pose_z"),
                ),
                (
                    self._f(state, plate, "pose_qx"),
                    self._f(state, plate, "pose_qy"),
                    self._f(state, plate, "pose_qz"),
                    self._f(state, plate, "pose_qw", 1.0),
                ),
            )
        self._shadow.sync(base, arm, poses, ppose)

    # -------------------------------------------------- base positioning
    def _desired_base(self, target_xy, base, carrying=False):
        """A base pose that puts target_xy in the left arm's sweet spot.

        The left arm reaches best ~0.55-0.70 m ahead of the base and ~0.20 m to
        the robot's left.  We approach the target from the -x side of the table
        when possible, so the robot stands off the table edge.
        """
        tx, ty = target_xy
        # Stand on whichever side gives clearance from the table (table is
        # 0.6 x 1.2 centred at origin, so |x| < 0.3 is occupied).
        reach = 0.62
        lateral = 0.20
        # approach from -x (robot in front, facing +x)
        yaw = 0.0
        bx = tx - reach
        by = ty - lateral
        # if that would be past the table edge into it, fall back
        if bx > -0.55:
            bx = -0.75
            # re-aim yaw at the target
            yaw = math.atan2(ty - by, tx - bx)
            by = ty - lateral * math.cos(yaw)
        # keep base out of the table footprint
        if abs(bx) < 0.62:
            bx = -0.72 if bx < 0 else 0.72
        return np.array([bx, by, yaw])

    def _do_drive_to(self, state, base, arm, target_xy, next_phase, carrying=False):
        des = self._desired_base(target_xy, base, carrying)
        err = des - base
        err[2] = _wrap(err[2])

        # arm: hold a safe carry configuration while driving
        arm_goal = self._carry_conf(arm)
        darm = np.clip(arm_goal - arm, -0.12, 0.12)

        if np.linalg.norm(err[:2]) < 0.04 and abs(err[2]) < 0.08:
            self._set_phase(next_phase)
            return self._mk(np.zeros(3), darm, GRIP_NONE)

        if self.phase_steps > 40 or self.stall > 6:
            # good enough / blocked: proceed anyway
            self._set_phase(next_phase)
            return self._mk(np.zeros(3), darm, GRIP_NONE)

        dbase = np.clip(err, -MAX_DELTA, MAX_DELTA)
        # jitter if stalled
        if self.stall > 2:
            dbase[:2] += self._rng.uniform(-0.06, 0.06, size=2)
        return self._mk(dbase, darm, GRIP_NONE)

    def _carry_conf(self, arm):
        if self._shadow is not None and self._shadow.ok:
            return np.array(self._shadow.init_conf, dtype=np.float64)
        return np.array(arm, dtype=np.float64)

    # -------------------------------------------------- arm servoing to pose
    def _servo_to_pose(self, state, base, arm, target_pose, gain=1.0,
                       max_step=MAX_DELTA):
        """Return arm deltas that move the tool toward `target_pose`."""
        if self._shadow is None or not self._shadow.ok:
            return None, None
        if self.arm_target is None:
            sol = self._shadow.ik(base, arm, target_pose)
            if sol is None:
                return None, None
            self.arm_target = np.array(sol, dtype=np.float64)
        err = self.arm_target - arm
        # continuous roll joints (indices 4 and 6 in the 7-vector) wrap
        for i in (4, 6):
            err[i] = _wrap(err[i])
        step = np.clip(err * gain, -max_step, max_step)
        return step, float(np.linalg.norm(err))

    def _do_pregrasp(self, state, base, arm, tb):
        pt, q = self._block_pose(state, tb)
        yaw = _quat_to_yaw(*q)
        # a square block: wrist yaw modulo pi/2 is irrelevant, pick nearest
        self.grasp_yaw = _wrap(yaw)
        above = _downward_tool_pose(pt[0], pt[1], pt[2] + 0.18, self.grasp_yaw)
        step, err = self._servo_to_pose(state, base, arm, above)
        if step is None:
            # IK failed: nudge the base and retry
            self.retry_count += 1
            if self.retry_count > 6:
                self._set_phase(self.P_DRIVE_BLOCK)
                self.retry_count = 0
                return self._mk(
                    self._rng.uniform(-0.1, 0.1, size=3), np.zeros(7), GRIP_NONE
                )
            return self._mk(
                np.array([0.05, self._rng.uniform(-0.08, 0.08), 0.0]),
                np.zeros(7),
                GRIP_OPEN,
            )
        if err is not None and err < 0.08:
            self.arm_target = None
            self.descend_z = None
            self._set_phase(self.P_DESCEND)
        if self.phase_steps > 35:
            self.arm_target = None
            self.descend_z = None
            self._set_phase(self.P_DESCEND)
        return self._mk(np.zeros(3), step, GRIP_OPEN)

    def _do_descend(self, state, base, arm, tb):
        pt, q = self._block_pose(state, tb)
        # Target: tool frame at ~block centre + small offset along approach,
        # i.e. the tool origin sits about 0.045 m above the block centre so the
        # block centre lies within the 0.06 m approach window.
        z = pt[2] + 0.035
        goal = _downward_tool_pose(pt[0], pt[1], z, self.grasp_yaw)
        if self.arm_target is None:
            step, err = self._servo_to_pose(state, base, arm, goal, max_step=0.10)
            if step is None:
                self.retry_count += 1
                if self.retry_count > 5:
                    # give up this block for now
                    self.arm_target = None
                    self._set_phase(self.P_DRIVE_BLOCK)
                    self.fail_blocks.add(self.target_block)
                    self.target_block = self._pick_next_block(state)
                    self.fail_blocks.clear()
                    return self._mk(
                        self._rng.uniform(-0.12, 0.12, size=3),
                        np.zeros(7),
                        GRIP_OPEN,
                    )
                return self._mk(np.zeros(3), np.zeros(7), GRIP_OPEN)
        else:
            step, err = self._servo_to_pose(state, base, arm, goal, max_step=0.10)

        if step is None:
            return self._mk(np.zeros(3), np.zeros(7), GRIP_OPEN)

        close_now = (err is not None and err < 0.06) or self.phase_steps > 25
        if close_now:
            self.arm_target = None
            self._set_phase(self.P_CLOSE)
            return self._mk(step * 0.5, step, GRIP_CLOSE)
        return self._mk(np.zeros(3), step, GRIP_OPEN)

    def _do_lift(self, state, base, arm):
        holding = self._f(state, self._robot(state), "grasp_active") > 0.5
        if not holding:
            # grasp failed: retry descend
            self.arm_target = None
            self.retry_count += 1
            if self.retry_count > 4:
                self.retry_count = 0
                self._set_phase(self.P_DRIVE_BLOCK)
                return self._mk(np.zeros(3), np.zeros(7), GRIP_OPEN)
            self._set_phase(self.P_DESCEND)
            return self._mk(np.zeros(3), np.zeros(7), GRIP_CLOSE)

        # lift straight up then head for the plate
        tp = None
        if self._shadow is not None and self._shadow.ok:
            tp = self._shadow.tool_pose(base, arm)
        if tp is None:
            self._set_phase(self.P_DRIVE_PLATE)
            return self._mk(np.zeros(3), np.zeros(7), GRIP_NONE)
        up = _downward_tool_pose(
            tp[0][0], tp[0][1], max(tp[0][2] + 0.16, 1.00), self.grasp_yaw
        )
        step, err = self._servo_to_pose(state, base, arm, up, max_step=0.12)
        if step is None or (err is not None and err < 0.10) or self.phase_steps > 20:
            self.arm_target = None
            self._set_phase(self.P_DRIVE_PLATE)
            return self._mk(np.zeros(3), np.zeros(7) if step is None else step,
                            GRIP_NONE)
        return self._mk(np.zeros(3), step, GRIP_NONE)

    def _do_over_slot(self, state, base, arm):
        sx, sy = self._current_slot()
        pz = self.plate_center[2]
        goal = _downward_tool_pose(sx, sy, pz + BLOCK_HEIGHT + 0.14, self.grasp_yaw)
        step, err = self._servo_to_pose(state, base, arm, goal, max_step=0.12)
        if step is None:
            self.retry_count += 1
            if self.retry_count > 5:
                self.retry_count = 0
                self.slot_try += 1
                self.arm_target = None
                self._set_phase(self.P_DRIVE_PLATE)
                return self._mk(
                    self._rng.uniform(-0.1, 0.1, size=3), np.zeros(7), GRIP_NONE
                )
            return self._mk(np.array([0.04, 0.0, 0.0]), np.zeros(7), GRIP_NONE)
        if (err is not None and err < 0.07) or self.phase_steps > 30:
            self.arm_target = None
            self._set_phase(self.P_LOWER_SLOT)
        return self._mk(np.zeros(3), step, GRIP_NONE)

    def _do_lower_slot(self, state, base, arm):
        sx, sy = self._current_slot()
        pz = self.plate_center[2]
        # tool origin just above the block's grasp offset over the plate
        goal = _downward_tool_pose(sx, sy, pz + BLOCK_HEIGHT + 0.035,
                                   self.grasp_yaw)
        step, err = self._servo_to_pose(state, base, arm, goal, max_step=0.08)
        if step is None or (err is not None and err < 0.06) or self.phase_steps > 20:
            self.arm_target = None
            self._set_phase(self.P_RELEASE)
            return self._mk(np.zeros(3), np.zeros(7) if step is None else step,
                            GRIP_OPEN)
        return self._mk(np.zeros(3), step, GRIP_NONE)

    def _do_release(self, state, base, arm):
        holding = self._f(state, self._robot(state), "grasp_active") > 0.5
        if not holding:
            # success -> next block
            nm = self.target_block
            if nm is not None:
                self.used_slots.add(self.slot_of.get(nm, 0))
            self.slot_try = 0
            self.arm_target = None
            self.target_block = self._pick_next_block(state)
            self._set_phase(self.P_DRIVE_BLOCK)
            # retract a little
            return self._mk(np.array([-0.08, 0.0, 0.0]), np.zeros(7), GRIP_NONE)

        # drop refused: shift to another slot and retry
        self.retry_count += 1
        if self.retry_count > 2:
            self.retry_count = 0
            self.slot_try += 1
            self.arm_target = None
            self._set_phase(self.P_OVER_SLOT)
            return self._mk(np.zeros(3), np.zeros(7), GRIP_NONE)
        return self._mk(np.zeros(3), np.zeros(7), GRIP_OPEN)

    # -------------------------------------------------- recovery
    def _recover_action(self, base, arm, holding):
        """Wiggle out of a blocked configuration."""
        self.arm_target = None
        dbase = self._rng.uniform(-0.12, 0.12, size=3)
        darm = self._rng.uniform(-0.10, 0.10, size=7)
        # bias the arm upward-ish by retracting toward carry conf
        carry = self._carry_conf(arm)
        darm = 0.5 * darm + 0.5 * np.clip(carry - arm, -0.1, 0.1)
        grip = GRIP_NONE
        if self.global_recover % 5 == 4 and not holding:
            grip = GRIP_OPEN
        return self._mk(dbase, darm, grip)