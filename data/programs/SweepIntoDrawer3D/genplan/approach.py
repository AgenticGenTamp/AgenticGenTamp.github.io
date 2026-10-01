"""
GeneratedApproach for kinder/SweepIntoDrawer3D-o5-v0.

IMPORTANT HONESTY NOTE (see comments throughout):
------------------------------------------------
In the preceding turn I said I did *not* have a verified strategy for this
environment, only a hypothesis.  That is still true.  I have not been able to
read the task JSON (`tidybot-*.json`) or the `tidybot3d_sweep3D` bilevel models,
so I do not know:

  * the exact world-frame extent of the goal region,
  * which of the six island drawers is the target,
  * the drawer handle pose / required pull direction,
  * whether the goal-region z-range is lenient.

What follows is therefore a *best-effort scripted policy* implementing the
hypothesis I described: park the base beside the kitchen island, open an island
drawer by hooking the gripper on the handle and driving the base backwards, then
grasp the wide (23 cm) wiper and make one straight plow stroke through the
tightly-clustered cube pile toward the open drawer, followed by a few repeat
strokes to catch stragglers.

It is a heuristic open-loop-ish script with light feedback (it re-aims the sweep
at the current centroid of the un-sunk cubes).  I expect it to be *far* from a
guaranteed solver.  I have deliberately NOT implemented any state-teleportation
exploit (writing cube poses via set_state), because that would be exploiting the
harness interface rather than acting in the environment.

The code is defensive: every action is clipped into `action_space`, the phase
machine always terminates into a safe idle phase, and nothing raises on
unexpected observation shapes.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Observation layout (from the environment's documented observation table).
# ---------------------------------------------------------------------------

NUM_CUBES = 5
CUBE_STRIDE = 16          # x,y,z,qw,qx,qy,qz,vx,vy,vz,wx,wy,wz,bb_x,bb_y,bb_z
CUBE_BASE = 0             # cube_0 starts at index 0

# kitchen_island fixture pose
IDX_ISLAND_X = 96
IDX_ISLAND_Y = 97
IDX_ISLAND_Z = 98
IDX_ISLAND_QW = 99
IDX_ISLAND_QZ = 102

# island drawer slide positions (s0c0, s0c1, s0c2, s1c0, s1c1, s1c2)
IDX_ISLAND_DRAWERS = (103, 104, 105, 106, 107, 108)

# robot block
IDX_BASE_X = 125
IDX_BASE_Y = 126
IDX_BASE_ROT = 127
IDX_ARM_J1 = 128          # .. 134 inclusive
IDX_GRIPPER = 135
IDX_VEL_BASE_X = 136

# wiper_0
IDX_WIPER_X = 147
IDX_WIPER_Y = 148
IDX_WIPER_Z = 149
IDX_WIPER_QW = 150
IDX_WIPER_QZ = 153
IDX_WIPER_BB_X = 160
IDX_WIPER_BB_Y = 161

OBS_DIM = 163


def _safe_get(state, idx, default=0.0):
    """Read state[idx] without ever raising."""
    try:
        v = float(np.asarray(state).reshape(-1)[idx])
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _cube_slice(state, i):
    """Return (x, y, z) of cube i, or None on failure."""
    b = CUBE_BASE + i * CUBE_STRIDE
    try:
        arr = np.asarray(state).reshape(-1)
        x, y, z = float(arr[b]), float(arr[b + 1]), float(arr[b + 2])
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            return None
        return np.array([x, y, z], dtype=np.float64)
    except Exception:
        return None


def _quat_yaw(qw, qz):
    """Yaw from a (mostly) z-axis quaternion, matching kinder's quat_to_yaw."""
    return 2.0 * float(np.arctan2(qz, qw))


def _wrap(a):
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class GeneratedApproach:
    """Scripted sweep-into-drawer policy.

    Action layout (11 dims), per the env's action-space description:
        [0:3]  base pose  (x, y, theta)
        [3:10] arm joint targets (7)
        [10]   gripper position

    The TidyBot config used by kinder sets ``act_delta=True`` by default, so
    actions are interpreted as *deltas* on the current targets.  Because I could
    not confirm that for this specific variant, the policy is written to be
    reasonable under a delta interpretation: it emits small, bounded corrections
    toward a desired absolute configuration.  Under an absolute interpretation
    the same code degenerates to "move slowly toward the setpoint", which is at
    worst slow rather than divergent.
    """

    # ---- tunables -------------------------------------------------------
    BASE_GAIN = 0.6           # proportional gain on base position error
    YAW_GAIN = 0.8
    ARM_GAIN = 0.7
    MAX_BASE_STEP = 0.08      # metres per control step (delta cap)
    MAX_YAW_STEP = 0.15       # radians per control step
    MAX_ARM_STEP = 0.20       # radians per joint per control step

    # Nominal "home / carry" arm posture, taken from the observed reset pose in
    # the provided example initial states (joints 1..7).
    ARM_HOME = np.array(
        [0.0, -0.349066, 3.14159265, -2.54818058, 0.0, -0.87266463, 1.57079637],
        dtype=np.float64,
    )
    # A lower, more forward posture used for reaching the counter top / handle.
    ARM_REACH = np.array(
        [0.0, -0.10, 3.14159265, -2.10, 0.0, -1.15, 1.57079637],
        dtype=np.float64,
    )
    # Posture used while plowing: elbow extended so the held wiper rides the top.
    ARM_PLOW = np.array(
        [0.0, 0.05, 3.14159265, -1.95, 0.0, -1.30, 1.57079637],
        dtype=np.float64,
    )

    GRIP_OPEN = 0.0
    GRIP_CLOSED = 1.0

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives if primitives is not None else {}

        # Action-space bookkeeping (robust to unexpected shapes).
        try:
            self.act_dim = int(np.prod(action_space.shape))
        except Exception:
            self.act_dim = 11
        try:
            self.act_low = np.asarray(action_space.low, dtype=np.float64).reshape(-1)
            self.act_high = np.asarray(action_space.high, dtype=np.float64).reshape(-1)
        except Exception:
            self.act_low = -np.ones(self.act_dim)
            self.act_high = np.ones(self.act_dim)

        # Index map into the action vector; falls back gracefully if the action
        # space is not the expected 11-dim layout.
        if self.act_dim >= 11:
            self.i_base = slice(0, 3)
            self.i_arm = slice(3, 10)
            self.i_grip = 10
        else:
            self.i_base = slice(0, min(3, self.act_dim))
            self.i_arm = slice(min(3, self.act_dim), self.act_dim)
            self.i_grip = None

        self._reset_internal()

    # ------------------------------------------------------------------
    # episode lifecycle
    # ------------------------------------------------------------------
    def _reset_internal(self):
        self.t = 0
        self.phase = "INIT"
        self.phase_t = 0
        self.plan = None              # dict of cached geometry
        self.sweep_idx = 0
        self.sweep_dir = np.array([1.0, 0.0])
        self.sweep_start = None
        self.sweep_end = None
        self.target_drawer = None
        self.prev_drawer_open = 0.0

    def reset(self, state, info):
        self._reset_internal()
        self._build_plan(state)
        return None

    # ------------------------------------------------------------------
    # geometry / planning
    # ------------------------------------------------------------------
    def _build_plan(self, state):
        """Cache island geometry, cube cluster centroid, wiper pose.

        HONEST CAVEAT: the drawer-opening geometry here is guessed.  The island
        pose is read from the observation; the handle is assumed to sit on the
        island face nearest the cube cluster, at roughly counter-drawer height.
        If that guess is wrong the drawer will simply not open and the sweep
        becomes a no-op push across the counter.
        """
        arr = np.asarray(state, dtype=np.float64).reshape(-1)

        island = np.array(
            [
                _safe_get(arr, IDX_ISLAND_X, 0.5),
                _safe_get(arr, IDX_ISLAND_Y, 0.0),
                _safe_get(arr, IDX_ISLAND_Z, 0.0),
            ]
        )
        island_yaw = _quat_yaw(
            _safe_get(arr, IDX_ISLAND_QW, 1.0), _safe_get(arr, IDX_ISLAND_QZ, 0.0)
        )

        cubes = [c for c in (_cube_slice(arr, i) for i in range(NUM_CUBES)) if c is not None]
        if cubes:
            centroid = np.mean(np.stack(cubes, axis=0), axis=0)
        else:
            centroid = np.array([0.70, -0.08, 0.47])

        wiper = np.array(
            [
                _safe_get(arr, IDX_WIPER_X, 0.86),
                _safe_get(arr, IDX_WIPER_Y, -0.39),
                _safe_get(arr, IDX_WIPER_Z, 0.46),
            ]
        )

        base = np.array(
            [_safe_get(arr, IDX_BASE_X, 1.2), _safe_get(arr, IDX_BASE_Y, -0.08)]
        )

        # Sweep direction: from the cube cluster toward the island centre, i.e.
        # push the pile *off* the near edge of the counter and into a drawer
        # that has been pulled out on the robot's side.  This is the part of the
        # hypothesis I am least confident about.
        to_island = island[:2] - centroid[:2]
        n = np.linalg.norm(to_island)
        if n < 1e-6:
            sweep_dir = np.array([-1.0, 0.0])
        else:
            sweep_dir = to_island / n

        # Choose the island drawer whose slide value we will monitor.  Without
        # the task JSON we cannot know which one is the goal, so we pick the
        # middle-front drawer (s0c1) as a default and monitor all of them.
        self.target_drawer = IDX_ISLAND_DRAWERS[1]

        # Standoff pose: park the base on the cube side of the island, facing it.
        approach_dir = -sweep_dir                      # points away from island
        stand = centroid[:2] + approach_dir * 0.62
        face_yaw = float(np.arctan2(sweep_dir[1], sweep_dir[0]))

        # Wiper grasp standoff.
        wiper_stand = wiper[:2] + approach_dir * 0.55
        wiper_yaw = float(np.arctan2(-approach_dir[1], -approach_dir[0]))

        self.plan = {
            "island": island,
            "island_yaw": island_yaw,
            "centroid": centroid,
            "wiper": wiper,
            "sweep_dir": sweep_dir,
            "approach_dir": approach_dir,
            "stand": stand,
            "face_yaw": face_yaw,
            "wiper_stand": wiper_stand,
            "wiper_yaw": wiper_yaw,
            "base0": base,
        }
        self.sweep_dir = sweep_dir

    # ------------------------------------------------------------------
    # low-level action construction
    # ------------------------------------------------------------------
    def _make_action(self, state, base_xy_target=None, yaw_target=None,
                     arm_target=None, grip_target=None):
        """Build a bounded action that moves toward the requested setpoints."""
        arr = np.asarray(state, dtype=np.float64).reshape(-1)
        a = np.zeros(self.act_dim, dtype=np.float64)

        # --- base ---
        bx = _safe_get(arr, IDX_BASE_X, 0.0)
        by = _safe_get(arr, IDX_BASE_Y, 0.0)
        bth = _safe_get(arr, IDX_BASE_ROT, 0.0)

        dx = dy = dth = 0.0
        if base_xy_target is not None:
            dx = float(base_xy_target[0]) - bx
            dy = float(base_xy_target[1]) - by
            dx = np.clip(self.BASE_GAIN * dx, -self.MAX_BASE_STEP, self.MAX_BASE_STEP)
            dy = np.clip(self.BASE_GAIN * dy, -self.MAX_BASE_STEP, self.MAX_BASE_STEP)
        if yaw_target is not None:
            dth = _wrap(float(yaw_target) - bth)
            dth = np.clip(self.YAW_GAIN * dth, -self.MAX_YAW_STEP, self.MAX_YAW_STEP)

        nb = self.i_base.stop - self.i_base.start
        vals = [dx, dy, dth][:nb]
        a[self.i_base.start:self.i_base.start + len(vals)] = vals

        # --- arm ---
        if arm_target is not None and self.i_arm.stop > self.i_arm.start:
            cur = np.array(
                [_safe_get(arr, IDX_ARM_J1 + k, 0.0) for k in range(7)],
                dtype=np.float64,
            )
            tgt = np.asarray(arm_target, dtype=np.float64).reshape(-1)
            if tgt.size < 7:
                tgt = np.concatenate([tgt, cur[tgt.size:]])
            d = self.ARM_GAIN * (tgt[:7] - cur)
            d = np.clip(d, -self.MAX_ARM_STEP, self.MAX_ARM_STEP)
            na = self.i_arm.stop - self.i_arm.start
            a[self.i_arm.start:self.i_arm.stop] = d[:na]

        # --- gripper ---
        if grip_target is not None and self.i_grip is not None:
            cur_g = _safe_get(arr, IDX_GRIPPER, 0.0)
            dg = np.clip(0.5 * (float(grip_target) - cur_g), -0.25, 0.25)
            a[self.i_grip] = dg

        # Final safety: clip into the declared action space.
        a = np.clip(a, self.act_low[:self.act_dim], self.act_high[:self.act_dim])
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        return a.astype(np.float32)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _max_island_drawer_open(self, state):
        arr = np.asarray(state, dtype=np.float64).reshape(-1)
        vals = [_safe_get(arr, i, 0.0) for i in IDX_ISLAND_DRAWERS]
        return max(vals) if vals else 0.0

    def _live_cubes(self, state):
        """Cubes still up on the counter (z above a rough drawer-lip threshold)."""
        arr = np.asarray(state, dtype=np.float64).reshape(-1)
        out = []
        for i in range(NUM_CUBES):
            c = _cube_slice(arr, i)
            if c is None:
                continue
            out.append(c)
        return out

    def _cluster_centroid(self, state):
        cubes = self._live_cubes(state)
        if not cubes:
            return self.plan["centroid"] if self.plan else np.array([0.7, -0.08, 0.47])
        # Use the highest cubes (still on the counter) to aim the plow.
        zs = np.array([c[2] for c in cubes])
        keep = [c for c, z in zip(cubes, zs) if z > (zs.max() - 0.08)]
        return np.mean(np.stack(keep, axis=0), axis=0)

    def _advance(self, name):
        self.phase = name
        self.phase_t = 0

    # ------------------------------------------------------------------
    # main policy
    # ------------------------------------------------------------------
    def get_action(self, state):
        self.t += 1
        self.phase_t += 1

        try:
            return self._policy(state)
        except Exception:
            # Never crash the rollout; emit a zero (no-op) action instead.
            return np.zeros(self.act_dim, dtype=np.float32)

    def _policy(self, state):
        if self.plan is None:
            self._build_plan(state)
        P = self.plan

        arr = np.asarray(state, dtype=np.float64).reshape(-1)
        drawer_open = self._max_island_drawer_open(state)

        # ---------------- phase machine ----------------
        if self.phase == "INIT":
            if self.phase_t > 5:
                self._advance("GOTO_DRAWER")
            return self._make_action(state, arm_target=self.ARM_HOME,
                                     grip_target=self.GRIP_OPEN)

        # --- 1. drive to the island face, gripper open, arm low ---
        if self.phase == "GOTO_DRAWER":
            tgt = P["centroid"][:2] + P["approach_dir"] * 0.58
            err = np.linalg.norm(np.array([_safe_get(arr, IDX_BASE_X),
                                           _safe_get(arr, IDX_BASE_Y)]) - tgt)
            if err < 0.07 or self.phase_t > 140:
                self._advance("REACH_HANDLE")
            return self._make_action(state, base_xy_target=tgt,
                                     yaw_target=P["face_yaw"],
                                     arm_target=self.ARM_REACH,
                                     grip_target=self.GRIP_OPEN)

        # --- 2. push in toward the drawer face so the gripper straddles a handle ---
        if self.phase == "REACH_HANDLE":
            tgt = P["centroid"][:2] + P["approach_dir"] * 0.42
            if self.phase_t > 60:
                self._advance("GRIP_HANDLE")
            return self._make_action(state, base_xy_target=tgt,
                                     yaw_target=P["face_yaw"],
                                     arm_target=self.ARM_REACH,
                                     grip_target=self.GRIP_OPEN)

        # --- 3. close on the handle ---
        if self.phase == "GRIP_HANDLE":
            if self.phase_t > 25:
                self._advance("PULL_DRAWER")
            return self._make_action(state, yaw_target=P["face_yaw"],
                                     arm_target=self.ARM_REACH,
                                     grip_target=self.GRIP_CLOSED)

        # --- 4. back the base up: the whole robot becomes the pulling actuator ---
        if self.phase == "PULL_DRAWER":
            tgt = P["centroid"][:2] + P["approach_dir"] * 0.95
            enough = drawer_open > 0.18
            if enough or self.phase_t > 150:
                self.prev_drawer_open = drawer_open
                self._advance("RELEASE_HANDLE")
            return self._make_action(state, base_xy_target=tgt,
                                     yaw_target=P["face_yaw"],
                                     arm_target=self.ARM_REACH,
                                     grip_target=self.GRIP_CLOSED)

        if self.phase == "RELEASE_HANDLE":
            if self.phase_t > 20:
                self._advance("GOTO_WIPER")
            return self._make_action(state, arm_target=self.ARM_HOME,
                                     grip_target=self.GRIP_OPEN)

        # --- 5. go get the wiper ---
        if self.phase == "GOTO_WIPER":
            w = np.array([_safe_get(arr, IDX_WIPER_X, P["wiper"][0]),
                          _safe_get(arr, IDX_WIPER_Y, P["wiper"][1])])
            tgt = w + P["approach_dir"] * 0.50
            err = np.linalg.norm(np.array([_safe_get(arr, IDX_BASE_X),
                                           _safe_get(arr, IDX_BASE_Y)]) - tgt)
            if err < 0.07 or self.phase_t > 160:
                self._advance("GRASP_WIPER")
            yaw = float(np.arctan2(-P["approach_dir"][1], -P["approach_dir"][0]))
            return self._make_action(state, base_xy_target=tgt, yaw_target=yaw,
                                     arm_target=self.ARM_REACH,
                                     grip_target=self.GRIP_OPEN)

        if self.phase == "GRASP_WIPER":
            w = np.array([_safe_get(arr, IDX_WIPER_X, P["wiper"][0]),
                          _safe_get(arr, IDX_WIPER_Y, P["wiper"][1])])
            tgt = w + P["approach_dir"] * 0.34
            yaw = float(np.arctan2(-P["approach_dir"][1], -P["approach_dir"][0]))
            if self.phase_t > 70:
                self._advance("CLOSE_WIPER")
            return self._make_action(state, base_xy_target=tgt, yaw_target=yaw,
                                     arm_target=self.ARM_REACH,
                                     grip_target=self.GRIP_OPEN)

        if self.phase == "CLOSE_WIPER":
            if self.phase_t > 25:
                self._setup_sweep(state)
                self._advance("SWEEP_APPROACH")
            return self._make_action(state, arm_target=self.ARM_REACH,
                                     grip_target=self.GRIP_CLOSED)

        # --- 6. repeated plow strokes ---
        if self.phase == "SWEEP_APPROACH":
            if self.sweep_start is None:
                self._setup_sweep(state)
            err = np.linalg.norm(np.array([_safe_get(arr, IDX_BASE_X),
                                           _safe_get(arr, IDX_BASE_Y)]) - self.sweep_start)
            if err < 0.08 or self.phase_t > 140:
                self._advance("SWEEP_PUSH")
            yaw = float(np.arctan2(self.sweep_dir[1], self.sweep_dir[0]))
            return self._make_action(state, base_xy_target=self.sweep_start,
                                     yaw_target=yaw,
                                     arm_target=self.ARM_PLOW,
                                     grip_target=self.GRIP_CLOSED)

        if self.phase == "SWEEP_PUSH":
            yaw = float(np.arctan2(self.sweep_dir[1], self.sweep_dir[0]))
            err = np.linalg.norm(np.array([_safe_get(arr, IDX_BASE_X),
                                           _safe_get(arr, IDX_BASE_Y)]) - self.sweep_end)
            if err < 0.06 or self.phase_t > 160:
                self.sweep_idx += 1
                if self.sweep_idx >= 6:
                    self._advance("IDLE")
                else:
                    self._setup_sweep(state)
                    self._advance("SWEEP_APPROACH")
            return self._make_action(state, base_xy_target=self.sweep_end,
                                     yaw_target=yaw,
                                     arm_target=self.ARM_PLOW,
                                     grip_target=self.GRIP_CLOSED)

        # --- terminal: hold still, keep emitting legal zero actions ---
        return self._make_action(state, arm_target=self.ARM_PLOW,
                                 grip_target=self.GRIP_CLOSED)

    # ------------------------------------------------------------------
    def _setup_sweep(self, state):
        """Re-aim the next plow stroke at the current cube centroid."""
        P = self.plan
        c = self._cluster_centroid(state)[:2]

        d = self.sweep_dir
        # Alternate a small lateral offset so successive strokes catch cubes
        # that squirted sideways out of the previous stroke.
        perp = np.array([-d[1], d[0]])
        lateral = ((self.sweep_idx % 3) - 1) * 0.09

        self.sweep_start = c - d * 0.50 + perp * lateral
        self.sweep_end = c + d * 0.34 + perp * lateral