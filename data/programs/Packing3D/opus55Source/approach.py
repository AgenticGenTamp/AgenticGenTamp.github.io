"""Scripted packing approach using a twin simulation for IK + verification."""
from __future__ import annotations

import itertools
import math
import time

import numpy as np

from layout import compute_layout
try:
    from layouts_cache import LAYOUTS
except Exception:  # pragma: no cover
    LAYOUTS = {}
try:
    from layouts_cache import LAYOUTS_ALT
except Exception:  # pragma: no cover
    LAYOUTS_ALT = {}
import os as _os
if _os.environ.get("MIRROR_LAYOUTS") and False:
    import json as _json
    LAYOUTS = dict(LAYOUTS)
    LAYOUTS.update(_json.load(open(_os.path.join(_os.path.dirname(__file__), "layouts_mirror.json"))))
from planner import PlanFailure, TwinPlanner, yaw_quat, ee_yaw
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import pybullet_inverse_kinematics
from planner import jac_ik
from blas1 import limit_blas_threads
limit_blas_threads(1)
from pybullet_helpers.joint import get_jointwise_difference

SAFE_Z = 0.30
PLACE_GAP = 0.003
GRASP_DZ = 0.07
DEBUG = False
VARIANT_BUDGET = 16.0
USE_QUICK_CHECK = True
COARSE_RES = 0.015
FINE_RES = 0.005


def _part_specs(state):
    parts = sorted([o for o in state if o.name.startswith("part")],
                   key=lambda o: int(o.name[4:]) if o.name[4:].isdigit() else 0)
    specs = []
    for o in parts:
        if o.type.name == "Kinematic3DCuboid":
            specs.append(("cuboid", (state.get(o, "half_extent_x"),
                                     state.get(o, "half_extent_y"))))
        else:
            tt = "equilateral" if int(round(state.get(o, "triangle_type"))) == 0 else "right"
            specs.append(("tri", (tt, state.get(o, "side_a"), state.get(o, "side_b"))))
    return parts, specs


def _period(spec):
    kind, p = spec
    if kind == "cuboid":
        return math.pi if abs(p[0] - p[1]) > 1e-9 else math.pi / 2
    if p[0] == "equilateral":
        return 2 * math.pi / 3
    return 2 * math.pi


def _sym_yaw(spec, target_yaw, cur_yaw):
    per = _period(spec)
    k = round((cur_yaw - target_yaw) / per)
    return target_yaw + k * per


def _part_yaw(state, name):
    o = state.get_object_from_name(name)
    qz, qw = state.get(o, "pose_qz"), state.get(o, "pose_qw")
    return 2 * math.atan2(qz, qw)


def _compatible(a, b):
    if a[0] != b[0]:
        return False
    return all((x == y) if isinstance(x, str) else abs(x - y) < 1e-6
               for x, y in zip(a[1], b[1]))


def _desc(spec):
    kind, p = spec
    if kind == "cuboid":
        if abs(p[0] - 0.05) < 1e-6 and abs(p[1] - 0.05) < 1e-6:
            return "S"
        return None
    if abs(p[1] - 0.1) < 1e-6 and abs(p[2] - 0.1) < 1e-6:
        return "E" if p[0] == "equilateral" else "R"
    return None


def _get_layout(specs, inner, rack):
    """Return (slot poses, slot specs)."""
    rx, ry, hx, hy = rack
    descs = [_desc(sp) for sp in specs]
    std_rack = (abs(rx - 0.3) < 1e-6 and abs(ry) < 1e-6 and abs(hx - 0.1) < 1e-6
                and abs(hy - 0.15) < 1e-6)
    if std_rack and all(d is not None for d in descs):
        key = "".join(sorted(descs))
        if key in LAYOUTS:
            rep = {d: sp for d, sp in zip(descs, specs)}
            slots = [tuple(v) for v in LAYOUTS[key]]
            alts = [[tuple(v) for v in LAYOUTS_ALT[key]]] if key in LAYOUTS_ALT else []
            return slots, [rep[c] for c in key], alts
    lay, _ = compute_layout(specs, inner)
    return lay, list(specs), []



def _footprint_of(spec):
    from layout import local_footprint
    kind, p = spec
    if kind == "cuboid":
        return local_footprint("cuboid", (p[0], p[1]))
    return local_footprint("tri", p)


def _reflect_layout(slots, slot_specs, cx, cy, fx, fy):
    """Reflect a layout across x=cx (fx) and/or y=cy (fy). Valid since all
    footprints are mirror-symmetric; we recover (origin, yaw) per slot."""
    out = []
    sgn = -1.0 if (fx != fy) else 1.0
    for (x, y, yaw), sp in zip(slots, slot_specs):
        fp = np.array(_footprint_of(sp).exterior.coords)[:-1]
        c, s_ = math.cos(yaw), math.sin(yaw)
        V = fp @ np.array([[c, s_], [-s_, c]]) + np.array([x, y])
        if fx:
            V[:, 0] = 2 * cx - V[:, 0]
        if fy:
            V[:, 1] = 2 * cy - V[:, 1]
        ox = 2 * cx - x if fx else x
        oy = 2 * cy - y if fy else y
        best = None
        for m in range(24):
            yw = sgn * yaw + m * math.pi / 12
            c, s_ = math.cos(yw), math.sin(yw)
            W = fp @ np.array([[c, s_], [-s_, c]]) + np.array([ox, oy])
            err = max(np.min(np.linalg.norm(V - w, axis=1)) for w in W)
            if best is None or err < best[0]:
                best = (err, yw)
        if best[0] > 1e-4:
            return None
        out.append((ox, oy, math.remainder(best[1], 2 * math.pi)))
    return out



def _single_part_variants(spec, state, name, inner, k=4):
    """Layouts for a lone part: keep its current yaw, place near it."""
    from shapely.geometry import box
    from layout import place_poly
    fp = _footprint_of(spec)
    lc = np.array(fp.centroid.coords[0])
    yaw = _part_yaw(state, name)
    o = state.get_object_from_name(name)
    px, py = state.get(o, "pose_x"), state.get(o, "pose_y")
    xmin, ymin, xmax, ymax = inner
    ok_region = box(xmin + 0.008, ymin + 0.008, xmax - 0.008, ymax - 0.008)
    c, s_ = math.cos(yaw), math.sin(yaw)
    cands = []
    for cx in np.linspace(xmin, xmax, 13):
        for cy in np.linspace(ymin, ymax, 19):
            ox = cx - (c * lc[0] - s_ * lc[1])
            oy = cy - (s_ * lc[0] + c * lc[1])
            if ok_region.contains(place_poly(fp, ox, oy, yaw)):
                cands.append((math.hypot(cx - px, cy - py), (float(ox), float(oy), float(yaw))))
    cands.sort(key=lambda t: t[0])
    out = []
    for d, sl in cands:
        if all(math.hypot(sl[0] - u[0], sl[1] - u[1]) > 0.03 for u in out):
            out.append(sl)
        if len(out) >= k:
            break
    return [[sl] for sl in out]


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self._twins = {}
        self._robot_type = None
        self._t0 = time.time()
        self._replans = 0
        self._expected = []
        self._plan = []
        self._idx = 0

    # ------------------------------------------------------------------
    def _twin(self, n):
        if n not in self._twins:
            self._twins[n] = TwinPlanner(n)
        return self._twins[n]

    def reset(self, state, info):
        self._t0 = time.time()
        self._plan = []
        self._idx = 0
        self._layout = None
        self._expected = []
        self._slot_spec = None
        self._slot_of = {}
        self._replans = 0
        try:
            self._make_plan(state)
        except Exception:  # pragma: no cover
            self._plan = []

    def _make_plan(self, state, check_res=COARSE_RES):
        parts, specs = _part_specs(state)
        n = len(parts)
        if n == 0:
            return
        tp = self._twin(n)
        tp.set_check_res(check_res)
        rack = state.get_object_from_name("rack")
        rx, ry = state.get(rack, "pose_x"), state.get(rack, "pose_y")
        hx, hy = state.get(rack, "half_extent_x"), state.get(rack, "half_extent_y")
        wall = 0.01
        inner = (rx - hx + wall, ry - hy + wall, rx + hx - wall, ry + hy - wall)
        self._floor_z = state.get(rack, "pose_z") - state.get(rack, "half_extent_z") + wall
        names = [o.name for o in parts]
        fast, complete = None, False
        if self._layout is None:
            base_lay, self._slot_spec, alts = _get_layout(specs, inner, (rx, ry, hx, hy))
            cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
            fams = []
            for L in [base_lay] + alts:
                fam = [L]
                for fx, fy in ((False, True), (True, False), (True, True)):
                    try:
                        v = _reflect_layout(L, self._slot_spec, cx, cy, fx, fy)
                    except Exception:
                        v = None
                    if v is not None:
                        fam.append(v)
                fams.append(fam)
            variants = []
            if n == 1:
                try:
                    variants.extend(_single_part_variants(self._slot_spec[0], state, names[0], inner))
                except Exception:
                    pass
            for k in range(max(len(f) for f in fams)):
                for f in fams:
                    if k < len(f):
                        variants.append(f[k])
            best = None
            t_first = None
            for vi, lay in enumerate(variants):
                el = time.time() - self._t0
                if vi > 0 and (el + 1.2 * t_first > VARIANT_BUDGET):
                    break
                tv = time.time()
                self._slot_of = {}
                try:
                    acts, comp = self._fast_plan(
                        state, names, specs, lay,
                        bound=(len(best[0]) if best else None),
                        deadline=(None if vi == 0 else self._t0 + VARIANT_BUDGET))
                except PlanFailure:
                    acts, comp = None, False
                if vi == 0:
                    t_first = time.time() - tv
                    self._layout = lay
                    v0 = (acts, comp, list(tp.expected), dict(self._slot_of))
                if DEBUG: print("variant", vi, comp, len(acts) if acts else None, round(time.time() - tv, 1))
                if comp and (best is None or len(acts) < len(best[0])):
                    best = (acts, list(tp.expected), dict(self._slot_of), lay)
            # extra diversity: re-run best layout with stage-1 candidates skipped
            skip = 1
            while best is not None and skip <= 6:
                el = time.time() - self._t0
                if el + 1.2 * t_first > VARIANT_BUDGET:
                    break
                self._slot_of = {}
                try:
                    acts, comp = self._fast_plan(
                        state, names, specs, best[3], bound=len(best[0]),
                        deadline=self._t0 + VARIANT_BUDGET, first_skip=skip)
                except PlanFailure:
                    acts, comp = None, False
                if DEBUG: print("skip", skip, comp, len(acts) if acts else None)
                if comp and len(acts) < len(best[0]):
                    best = (acts, list(tp.expected), dict(self._slot_of), best[3])
                skip += 1
            if best is not None:
                self._plan = best[0]
                self._expected = best[1]
                self._slot_of = best[2]
                self._layout = best[3]
                self._idx = 0
                return
            fast, complete = v0[0], v0[1]
            tp.expected = v0[2]
            self._slot_of = v0[3]
        else:
            try:
                fast, complete = self._fast_plan(state, names, specs, self._layout)
            except PlanFailure:
                fast, complete = None, False
        if fast is not None and complete:
            self._plan = fast
            self._expected = list(tp.expected)
            self._idx = 0
            return
        prefix = list(fast) if fast is not None else []
        prefix_exp = list(tp.expected) if fast is not None else []
        if fast is None:
            tp.set_state(state)
        # continue from the twin's current state with the robust method
        base_snap = tp.env.get_state()
        targets = {}
        used = set(self._slot_of.values())
        for nm, sp in zip(names, specs):
            si = self._slot_of.get(nm)
            if si is None:
                for j, ss in enumerate(self._slot_spec):
                    if j not in used and _compatible(sp, ss):
                        si = j
                        break
            if si is None:
                si = 0
            used.add(si)
            targets[nm] = self._layout[si]
        best = None
        yaw_opts = [0.0, math.pi / 2, -math.pi / 2, math.pi]
        for attempt in range(12):
            if time.time() - self._t0 > 52:
                break
            tp.env.set_state(base_snap)
            tp.state = tp.env.get_state()
            tp.actions = []
            tp.expected = []
            todo = [nm for nm in names if not self._is_placed(tp, nm)]
            rng = np.random.default_rng(attempt)
            if attempt > 0:
                rng.shuffle(todo)
            ok = True
            try:
                self._release_if_held(tp)
                for nm in todo:
                    placed = False
                    for gy in (yaw_opts if attempt == 0 else rng.permutation(yaw_opts)):
                        snap = tp.snapshot()
                        try:
                            self._pick_place(tp, nm, targets[nm], gy)
                            placed = True
                            break
                        except PlanFailure:
                            tp.restore(snap)
                    if not placed:
                        ok = False
                        break
            except PlanFailure:
                ok = False
            if ok:
                best = (list(tp.actions), list(tp.expected))
                break
            if best is None or len(tp.actions) > len(best[0]):
                best = (list(tp.actions), list(tp.expected))
        self._plan = prefix + (best[0] if best else [])
        self._expected = prefix_exp + (best[1] if best else [])
        self._idx = 0

    def _is_placed(self, tp, name):
        try:
            return bool(tp.env._part_is_seated_in_rack(name, tp.state)) and \
                tp.env._grasped_object != name
        except Exception:
            return False

    def _release_if_held(self, tp):
        if tp.env._grasped_object is not None:
            tp.gripper(False)
            e = tp.ee()
            tp.move_cartesian(Pose((e.position[0], e.position[1], SAFE_Z), e.orientation))

    def _pick_place(self, tp, name, target, grasp_yaw):
        st = tp.state
        cen = st.get_object_pose(name).position
        ori = tp.down_yaw(grasp_yaw)
        tp.move_cartesian(Pose((cen[0], cen[1], SAFE_Z), ori))
        tp.move_cartesian(Pose((cen[0], cen[1], cen[2] + GRASP_DZ), ori))
        tp.gripper(True)
        if tp.env._grasped_object != name:
            raise PlanFailure("wrong grasp")
        tf = tp.env._grasped_object_transform
        tp.move_cartesian(Pose((cen[0], cen[1], SAFE_Z), ori))
        x, y, yaw = target
        tgt = Pose((x, y, self._floor_z + 0.01 + PLACE_GAP), yaw_quat(yaw))
        eet = multiply_poses(tgt, tf.invert())
        tp.move_cartesian(Pose((eet.position[0], eet.position[1], SAFE_Z), eet.orientation))
        tp.move_cartesian(eet, step_len=0.01)
        tp.gripper(False)
        if not self._is_placed(tp, name):
            raise PlanFailure("not seated")
        tp.move_cartesian(Pose((eet.position[0], eet.position[1], SAFE_Z), eet.orientation))


    # ---------------- fast (direct joint-space) primitives ----------------
    def _try_seqs(self, tp, seqs, check):
        """seqs: list of lists of (pose, grip). First successful wins."""
        for seq in seqs:
            snap = tp.snapshot()
            try:
                for pose, grip in seq:
                    tp.move_direct(pose, grip=grip)
                if check():
                    return True
            except PlanFailure:
                pass
            tp.restore(snap)
        return False

    def _fast_pick(self, tp, name, grasp_yaw):
        cen = tp.state.get_object_pose(name).position
        ori = tp.down_yaw(grasp_yaw)
        e = tp.ee()
        g = Pose((cen[0], cen[1], cen[2] + GRASP_DZ), ori)
        pre = Pose((cen[0], cen[1], cen[2] + GRASP_DZ + 0.06), ori)
        up = Pose((e.position[0], e.position[1], max(e.position[2], SAFE_Z)), e.orientation)
        hi = Pose((cen[0], cen[1], SAFE_Z), ori)
        seqs = [[(g, -1.0)], [(pre, None), (g, -1.0)],
                [(up, None), (hi, None), (g, -1.0)]]
        return self._try_seqs(tp, seqs, lambda: tp.env._grasped_object == name)

    def _fast_place(self, tp, name, target):
        tf = tp.env._grasped_object_transform
        x, y, yaw = target
        tgt = Pose((x, y, self._floor_z + 0.01 + PLACE_GAP), yaw_quat(yaw))
        eet = multiply_poses(tgt, tf.invert())
        e = tp.ee()
        p = eet.position
        seqs = []
        seqs.append([(eet, 1.0)])
        for h in (0.04, 0.1):
            lift = Pose((e.position[0], e.position[1], e.position[2] + h), e.orientation)
            above = Pose((p[0], p[1], p[2] + h), eet.orientation)
            seqs.append([(above, None), (eet, 1.0)])
            seqs.append([(lift, None), (above, None), (eet, 1.0)])
        up = Pose((e.position[0], e.position[1], SAFE_Z), e.orientation)
        above = Pose((p[0], p[1], SAFE_Z), eet.orientation)
        near = Pose((p[0], p[1], p[2] + 0.02), eet.orientation)
        seqs.append([(up, None), (above, None), (near, None), (eet, 1.0)])
        return self._try_seqs(
            tp, seqs,
            lambda: tp.env._grasped_object is None and self._is_placed(tp, name))

    def _ik_from(self, tp, qseed, pose):
        full = list(tp.arm.get_joint_positions())
        try:
            tp.arm.set_joints(list(qseed) + full[7:])
            q = jac_ik(tp.arm, pose)
            return np.array(q[:7])
        except Exception:
            return None
        finally:
            tp.arm.set_joints(full)

    def _chain_cost(self, tp, q0, poses):
        """Estimated #steps to traverse poses by direct joint moves."""
        q = q0
        cost = 0
        for pose in poses:
            q1 = self._ik_from(tp, q, pose)
            if q1 is None:
                return None, None
            d = np.array(get_jointwise_difference(tp.jinfos, list(q1), list(q)))
            cost += max(1, int(math.ceil(float(np.max(np.abs(d))) / 0.2 - 1e-9)))
            q = q1
        return cost, q

    def _pick_options(self, tp, name, gy, e):
        cen = tp.state.get_object_pose(name).position
        ori = tp.down_yaw(gy)
        ez = e.position[2]
        g = Pose((cen[0], cen[1], cen[2] + GRASP_DZ), ori)
        pre = Pose((cen[0], cen[1], cen[2] + GRASP_DZ + 0.06), ori)
        up1 = Pose((e.position[0], e.position[1], ez + 0.06), e.orientation)
        up2 = Pose((e.position[0], e.position[1], ez + 0.10), e.orientation)
        hi2 = Pose((cen[0], cen[1], ez + 0.10), ori)
        up3 = Pose((e.position[0], e.position[1], max(ez, SAFE_Z)), e.orientation)
        hi3 = Pose((cen[0], cen[1], max(ez, SAFE_Z)), ori)
        return g, [[g], [pre, g], [up1, pre, g], [up2, hi2, pre, g],
                   [up3, hi3, pre, g]]

    def _place_options(self, eet, e):
        p = eet.position
        ez = e.position[2]
        def lift(h):
            return Pose((e.position[0], e.position[1], ez + h), e.orientation)
        def above(h):
            return Pose((p[0], p[1], p[2] + h), eet.orientation)
        near = above(0.02)
        sz = max(SAFE_Z, ez) - p[2]
        return [[eet], [above(0.07), eet], [lift(0.07), above(0.07), eet],
                [lift(0.11), above(0.11), eet],
                [lift(0.11), above(0.11), near, eet],
                [lift(sz), above(sz), near, eet]]

    def _via_options(self, e, t, fracs, heights):
        from planner import slerp
        p0 = np.array(e.position)
        p1 = np.array(t.position)
        q0 = np.array(e.orientation)
        q1 = np.array(t.orientation)
        if np.dot(q0, q1) < 0:
            q1 = -q1
        zb = max(p0[2], p1[2])
        opts = []
        for f in fracs:
            for h in heights:
                xy = p0[:2] + f * (p1[:2] - p0[:2])
                q = slerp(q0, q1, f)
                v = Pose((float(xy[0]), float(xy[1]), zb + h), tuple(float(c) for c in q))
                opts.append([v, t])
        return opts

    def _exec_seq(self, tp, poses, grip):
        for i, pose in enumerate(poses):
            tp.move_direct(pose, grip=grip if i == len(poses) - 1 else None)

    def _chain(self, tp, q0, poses):
        """IK chain -> (est cost, list of joint configs) or (None, None)."""
        qs = []
        q = q0
        cost = 0
        for pose in poses:
            q1 = self._ik_from(tp, q, pose)
            if q1 is None:
                return None, None
            d = np.array(get_jointwise_difference(tp.jinfos, list(q1), list(q)))
            m = float(np.max(np.abs(d)))
            cost += max(1, int(math.ceil(m / 0.2 - 1e-9)))
            q = q + d
            qs.append(q.copy())
        return cost, qs

    def _quick_check(self, tp, q0, qs, res=0.04):
        """Coarse collision check along straight joint segments, sampled
        coarse-to-fine so that collisions are found early."""
        if not USE_QUICK_CHECK:
            return True
        env = tp.env
        base = env.robot.get_base()
        fs = tp.arm.get_finger_state()
        full = list(tp.arm.get_joint_positions())
        pts = []
        q = np.array(q0)
        for q1 in qs:
            d = np.array(q1) - q
            k = max(1, int(math.ceil(float(np.max(np.abs(d))) / res)))
            pts.extend(q + d * i / k for i in range(1, k + 1))
            q = np.array(q1)
        n = len(pts)
        order = []
        seen = set()
        stride = 1
        while stride * 2 < n:
            stride *= 2
        while stride >= 1:
            for i in range(stride - 1, n, stride):
                if i not in seen:
                    seen.add(i)
                    order.append(i)
            stride //= 2
        ok = True
        try:
            for i in order:
                env._set_robot_and_held_object(base, list(pts[i]), fs)
                if env._robot_or_held_object_collision_exists():
                    ok = False
                    break
        finally:
            env._set_robot_and_held_object(base, full[:7], fs)
        return ok

    def _fast_plan(self, state, names, specs, slots, bound=None, deadline=None,
                   first_skip=0):
        """Greedy: each stage choose (part, slot, grasp yaw, waypoints) with
        fewest estimated steps, verified by simulation."""
        tp = self._twin(len(names))
        tp.set_state(state)
        tp.actions = []
        tp.expected = []
        kinds = {nm: sp for nm, sp in zip(names, specs)}
        free = list(range(len(slots)))
        if tp.env._grasped_object is not None:
            return None, False
        todo = [nm for nm in names if not self._is_placed(tp, nm)]
        for nm in names:
            if nm not in todo and self._slot_of.get(nm) in free:
                free.remove(self._slot_of[nm])
        place_z = self._floor_z + 0.01 + PLACE_GAP
        while todo:
            t_est = time.time()
            q0 = tp.cur_joints()
            e = tp.ee()
            g0 = ee_yaw(tp, e.orientation)
            combos = []
            for nm in todo:
                obj_pose = tp.state.get_object_origin_pose(nm)
                pyaw = _part_yaw(tp.state, nm)
                gys = {}
                for si in free:
                    if not _compatible(kinds[nm], self._slot_spec[si]):
                        continue
                    tyaw = _sym_yaw(kinds[nm], slots[si][2], pyaw)
                    delta = tyaw - pyaw
                    cand_g = [g0, g0 - delta / 2, g0 - delta, g0 + math.pi / 2 - delta / 2]
                    uniq = []
                    for gy in cand_g:
                        if all(abs(math.remainder(gy - u, 2 * math.pi)) > 0.1 for u in uniq):
                            uniq.append(gy)
                    for gy in uniq:
                        key = round(gy, 3)
                        if key not in gys:
                            g, popts = self._pick_options(tp, nm, gy, e)
                            pc = []
                            for pi_, pseq in enumerate(popts):
                                c, qs = self._chain(tp, q0, pseq)
                                if c is not None:
                                    pc.append((c, pi_, qs))
                            gys[key] = (g, sorted(pc, key=lambda x: x[0]))
                        g, pc = gys[key]
                        if not pc:
                            continue
                        qg = pc[0][2][-1]
                        tf_est = multiply_poses(g.invert(), obj_pose)
                        tgt = Pose((slots[si][0], slots[si][1], place_z), yaw_quat(tyaw))
                        eet = multiply_poses(tgt, tf_est.invert())
                        lc = []
                        for li, lseq in enumerate(self._place_options(eet, g)):
                            if li not in (0, 2):
                                continue
                            c, _ = self._chain(tp, qg, lseq)
                            if c is not None:
                                lc.append((c, li))
                        if not lc:
                            continue
                        lc.sort()
                        combos.append((pc[0][0] + lc[0][0], nm, si, gy, tyaw, pc, lc))
            combos.sort(key=lambda c: c[0])
            if first_skip and len(todo) == len(names):
                if len(combos) <= first_skip:
                    return None, False
                combos = combos[first_skip:]
            if DEBUG: print(" est time", round(time.time() - t_est, 2), "ncombos", len(combos))
            base = tp.snapshot()
            done = False
            for combo in combos:
                if time.time() - self._t0 > 50 or (deadline is not None and time.time() > deadline):
                    tp.restore(base)
                    return list(tp.actions), False
                _, nm, si, gy, tyaw, pc, lc = combo
                tp.restore(base)
                picked = False
                g_pose, popts_all = self._pick_options(tp, nm, gy, e)
                popts_all = popts_all + self._via_options(
                    e, g_pose, (0.0, 0.25, 0.5), (0.03, 0.06, 0.1))
                pc = []
                for pi_, pseq in enumerate(popts_all):
                    c, qs = self._chain(tp, q0, pseq)
                    if c is not None:
                        pc.append((c, pi_, qs))
                pc.sort(key=lambda x: x[0])
                for c, pi_, qs in pc:
                    if not self._quick_check(tp, q0, qs):
                        continue
                    try:
                        self._exec_seq(tp, popts_all[pi_], -1.0)
                        if tp.env._grasped_object == nm:
                            picked = True
                            break
                    except PlanFailure as ex:
                        if DEBUG: print("  pick fail", nm, pi_, ex)
                    tp.restore(base)
                if not picked:
                    continue
                after_pick = tp.snapshot()
                tf = tp.env._grasped_object_transform
                tgt = Pose((slots[si][0], slots[si][1], place_z), yaw_quat(tyaw))
                eet = multiply_poses(tgt, tf.invert())
                e2 = tp.ee()
                q1 = tp.cur_joints()
                lopts = self._place_options(eet, e2) + self._via_options(
                    e2, eet, (0.25, 0.5, 0.75), (0.04, 0.07, 0.1, 0.13))
                lc = []
                for li, lseq in enumerate(lopts):
                    c, qs = self._chain(tp, q1, lseq)
                    if c is not None:
                        lc.append((c, li, qs))
                lc.sort(key=lambda x: x[0])
                placed = False
                for c, li, qs in lc:
                    if not self._quick_check(tp, q1, qs):
                        continue
                    try:
                        self._exec_seq(tp, lopts[li], 1.0)
                        if tp.env._grasped_object is None and self._is_placed(tp, nm):
                            placed = True
                            break
                    except PlanFailure as ex:
                        if DEBUG: print("  place fail", nm, li, ex)
                    tp.restore(after_pick)
                if not placed:
                    continue
                if DEBUG: print("stage ok", nm, si, round(gy, 2), pi_, li, "est", combo[0], "actual", len(tp.actions) - base[1], "t", round(time.time() - self._t0, 1))
                done = True
                todo.remove(nm)
                if bound is not None and len(tp.actions) + 7 * len(todo) >= bound:
                    return list(tp.actions), False
                free.remove(si)
                self._slot_of[nm] = si
                break
            if not done:
                tp.restore(base)
                return list(tp.actions), False
        return list(tp.actions), True

    # ------------------------------------------------------------------
    def _robot_joints(self, state):
        r = None
        for o in state:
            if "Robot" in o.type.name:
                r = o
                break
        if r is None:
            return None
        return np.array([state.get(r, f"joint_{i}") for i in range(1, 8)])

    def _diverged(self, state):
        if self._idx == 0 or self._idx > len(self._expected):
            return False
        try:
            q = self._robot_joints(state)
        except Exception:
            return False
        if q is None:
            return False
        exp = self._expected[self._idx - 1]
        d = np.remainder(q - exp + math.pi, 2 * math.pi) - math.pi
        return float(np.max(np.abs(d))) > 1e-3

    def get_action(self, state):
        if self._diverged(state) and self._replans < 4 and time.time() - self._t0 < 52:
            self._replans += 1
            try:
                self._make_plan(state, check_res=FINE_RES)
            except Exception:
                self._plan = []
        if self._idx < len(self._plan):
            a = self._plan[self._idx]
            self._idx += 1
            return np.array(a, dtype=np.float32)
        # plan exhausted: try replanning from current state (limited);
        # an unfinished episode is a failure anyway, so ignore the clock here.
        if self._replans < 4:
            self._replans += 1
            try:
                self._make_plan(state, check_res=FINE_RES)
            except Exception:
                self._plan = []
            if self._idx < len(self._plan):
                a = self._plan[self._idx]
                self._idx += 1
                return np.array(a, dtype=np.float32)
        return np.zeros(11, dtype=np.float32)
