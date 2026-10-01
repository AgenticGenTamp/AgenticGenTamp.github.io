"""PR2Packed approach: exact FK + DLS IK, direct joint-space moves with
endpoint collision checking, close/open on arrival steps."""
import math
import os
import time
import numpy as np

from blas1 import limit_blas_threads

limit_blas_threads()

from kin import fk_full, ik_down, ik_refine, wrap, LO, HI, CONT, Q0, shoulder_xy, SX as SX_B, SY as SY_B
from geom import OBB, obb_overlap, seg_box, rect_overlap_2d
from gridroute import grid_path

STEP = 0.2
FUT_W = float(os.environ.get("PR2_FUT_W", "1.0"))
SHORTEN = os.environ.get("PR2_SHORTEN", "1") == "1"
REFINE = os.environ.get("PR2_REFINE", "1") == "1"
ROUTE_ALL = os.environ.get("PR2_ROUTE_ALL", "1") == "1"
LIFT_ROUTE = os.environ.get("PR2_LIFT_ROUTE", "1") == "1"
GRID = os.environ.get("PR2_GRID", "1") == "1"
ROLL_TOL = float(os.environ.get("PR2_ROLL_TOL", "0.24"))
TABLE_TOP = 0.731
BLOCK_HZ = 0.05
GRASP_DZ = 0.012          # tool z above block centre at grasp
PLATE_MARGIN = 0.006
NEAR_TOOL_Z_HIGH = 0.875  # tool height for placing when fingers must clear neighbours

# ---------------- base candidates ----------------
BASE_CANDS = []
for _y in (-0.22, -0.11, 0.0, 0.11, 0.22):
    BASE_CANDS.append((-0.47, _y, 0.0))
    BASE_CANDS.append((0.47, _y, math.pi))
for _x in (-0.3, -0.15, 0.0, 0.15, 0.3):
    BASE_CANDS.append((_x, -0.99, 0.0))
    BASE_CANDS.append((_x, 0.99, math.pi))
BASE_CANDS.append((-1.0, 0.0, 0.0))


def block_bases(x, y, dense=False):
    """Base poses putting the shoulder at good distances from (x, y)."""
    out = []
    nth = 24 if dense else 12
    ds = (0.4, 0.55, 0.7) if dense else (0.5, 0.65)
    for i in range(nth):
        th = -math.pi + 2 * math.pi * i / nth
        c, s = math.cos(th), math.sin(th)
        for d in ds:
            # shoulder at block - d * dir(th + small offset), base heading th
            shx, shy = x - d * c, y - d * s
            bx = shx - (c * SX_B - s * SY_B)
            by = shy - (s * SX_B + c * SY_B)
            b = (bx, by, th)
            if base_ok(b):
                out.append(b)
    return out


def base_ok(b):
    """Base footprint vs table (2D, conservative) with under-edge allowance."""
    x, y, t = b
    t = wrap(t)
    if abs(t) < 0.16:
        e = 0.36 * abs(math.sin(t))
        if -0.70 + e <= x <= -0.455 - e and abs(y) <= 0.23 - e:
            return True
    if abs(wrap(t - math.pi)) < 0.16:
        e = 0.36 * abs(math.sin(t))
        if 0.455 + e <= x <= 0.70 - e and abs(y) <= 0.23 - e:
            return True
    # base rect in base frame: x[-0.335,0.335], y[-0.36,0.36]
    c = (x + math.cos(t) * 0.0, y + math.sin(t) * 0.0)
    return not rect_overlap_2d(c, t, (0.335, 0.365), (0.0, 0.0), 0.0, (0.3, 0.6), margin=0.01)


def cfg_diff(a, b):
    """b - a with wrapping for continuous dims (cfg = 10 vec)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    d[2] = wrap(d[2])
    d[7] = wrap(d[7])
    d[9] = wrap(d[9])
    return d


def nsteps(a, b):
    m = np.abs(cfg_diff(a, b)).max()
    return int(math.ceil(m / STEP - 1e-3))


def yaw_of_quat(qx, qy, qz, qw):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


class World:
    """Static snapshot of blocks used for collision checks."""

    def __init__(self, blocks):
        # blocks: dict name -> (x, y, z, yaw)
        self.blocks = blocks
        self.boxes = {n: OBB((b[0], b[1], b[2]), Rz(b[3]), (0.035, 0.035, BLOCK_HZ))
                      for n, b in blocks.items()}
        self.table = OBB((0, 0, TABLE_TOP / 2), np.eye(3), (0.3, 0.6, TABLE_TOP / 2))


def _tbox(tool, R, x0, x1, y0, y1, zh):
    c = tool + R @ np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.0])
    return OBB(c, R, ((x1 - x0) / 2, (y1 - y0) / 2, zh))


def arm_boxes(pts, R, closed):
    """Collision boxes: [upper arm, forearm, palm/wrist, gripper parts...]."""
    shoulder, upper, elbow, wrist, tool = pts
    boxes = [seg_box(upper, elbow, 0.075), seg_box(elbow, wrist, 0.06),
             _tbox(tool, R, -0.20, -0.05, -0.068, 0.068, 0.03)]
    if closed:
        boxes.append(_tbox(tool, R, -0.10, 0.02, -0.054, 0.054, 0.029))
    else:
        for sg in (1.0, -1.0):
            boxes.append(_tbox(tool, R, -0.02, 0.005, *sorted((sg * 0.045, sg * 0.076)), 0.016))
            boxes.append(_tbox(tool, R, -0.10, -0.02, *sorted((sg * 0.038, sg * 0.092)), 0.016))
    return boxes


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives
        self.plan = []
        self.debug = False

    # ------------------------------------------------------------ parsing
    def _parse(self, state):
        robot = None; blocks = {}; plate = None
        for o in state.get_objects(self._type("robot")):
            robot = o
        for o in state.get_objects(self._type("surface")):
            if o.name == "plate":
                plate = o
        rf = lambda f: float(state.get(robot, f))
        cfg = np.array([rf("base_x"), rf("base_y"), rf("base_rot")] +
                       [rf("joint_%d" % i) for i in range(1, 8)])
        held = None
        for o in state.get_objects(self._type("block")):
            g = lambda f: float(state.get(o, f))
            yaw = yaw_of_quat(g("pose_qx"), g("pose_qy"), g("pose_qz"), g("pose_qw"))
            blocks[o.name] = (g("pose_x"), g("pose_y"), g("pose_z"), yaw)
            if g("grasp_active") > 0.5:
                held = o.name
        gtf = np.array([rf("grasp_tf_x"), rf("grasp_tf_y"), rf("grasp_tf_z")])
        gq = np.array([rf("grasp_tf_qx"), rf("grasp_tf_qy"), rf("grasp_tf_qz"), rf("grasp_tf_qw")])
        holding = rf("grasp_active") > 0.5
        if plate is not None:
            self.plate_c = (float(state.get(plate, "pose_x")), float(state.get(plate, "pose_y")))
            self.plate_h = (float(state.get(plate, "half_extent_x")), float(state.get(plate, "half_extent_y")))
        return cfg, blocks, holding, held, gtf, gq

    def _type(self, name):
        try:
            return self.observation_space.get_type(name)
        except Exception:
            for t in self.observation_space.types:
                if getattr(t, "name", None) == name:
                    return t
            raise

    # ------------------------------------------------------------ geometry
    def _on_plate(self, b, margin=0.0):
        x, y, z, yaw = b
        if z > 0.80 or z < 0.77:
            return False
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        ex = 0.035 * (c + s)
        px, py = self.plate_c; hx, hy = self.plate_h
        return abs(x - px) + ex <= hx - margin and abs(y - py) + ex <= hy - margin

    def _config_ok(self, cfg, world, exclude=None, held=None, closed=False, check_table_fingers=True,
                   margin=None):
        """held: (pos offset p_off in tool frame, R_rel) of held block relative to tool."""
        base = cfg[:3]
        if not base_ok(base):
            return False
        for bc in self.bad_cfgs:
            if np.abs(cfg_diff(bc, cfg)).max() < 0.03:
                return False
        for bb in self.bad_bases:
            if abs(bb[0] - base[0]) < 0.03 and abs(bb[1] - base[1]) < 0.03 and abs(wrap(bb[2] - base[2])) < 0.08:
                return False
        mg = self.margin if margin is None else margin
        pts, _, _, tool, R = fk_full(base, cfg[3:])
        boxes = arm_boxes(pts, R, closed)
        for i, bx in enumerate(boxes):
            if (i < 3 or check_table_fingers) and obb_overlap(bx, world.table, mg if i < 3 else 0.0):
                return False
            for n, ob in world.boxes.items():
                if n == exclude:
                    continue
                if np.linalg.norm(bx.c[:2] - ob.c[:2]) > 0.5:
                    continue
                if obb_overlap(bx, ob, mg):
                    return False
        if held is not None:
            po, Rrel = held
            hc = tool + R @ po
            if hc[2] - BLOCK_HZ < TABLE_TOP + 0.001 and abs(hc[0]) < 0.36 and abs(hc[1]) < 0.66:
                return False
            hb = OBB(hc, R @ Rrel, (0.035, 0.035, BLOCK_HZ))
            for n, ob in world.boxes.items():
                if n == exclude:
                    continue
                if obb_overlap(hb, ob, max(mg, 0.002)):
                    return False
        return True

    def _path_ok(self, a, b, world, exclude_final=None, held=None, closed=False):
        n = nsteps(a, b)
        d = cfg_diff(a, b)
        for k in range(1, n):
            c = a + d * (k / n)
            if not self._config_ok(c, world, None, held, closed):
                return False
        return True

    # ------------------------------------------------------------ planning
    def _ik_cfg(self, base, pos, q_init, yaw=None, p_off=None, multi=False):
        q, ok = ik_down(base, pos, q_init, yaw=yaw, p_off=p_off)
        if not ok:
            q, ok = ik_down(base, pos, q, yaw=yaw, p_off=p_off, iters=170)
        if not ok and multi:
            sh = shoulder_xy(base)
            pan = float(np.clip(wrap(math.atan2(pos[1] - sh[1], pos[0] - sh[0]) - base[2]), LO[0], HI[0]))
            inits = [np.array([pan, -0.5, 2.1, -2.0, 1.3, -0.85, 2.0]),
                     np.array([pan, Q0[1], Q0[2], Q0[3], Q0[4], Q0[5], Q0[6]]),
                     self.rng.uniform(np.maximum(LO, -3), np.minimum(HI, 3))]
            for qi in inits:
                q, ok = ik_down(base, pos, qi, yaw=yaw, p_off=p_off, iters=200)
                if ok:
                    break
        if not ok:
            return None
        return np.concatenate([np.asarray(base, float), q])

    def _lift(self, cfg, dz, p_off=None):
        pts, _, _, tool, R = fk_full(cfg[:3], cfg[3:])
        yaw = math.atan2(R[1, 1], R[0, 1])
        pos = tool + np.array([0, 0, dz])
        return self._ik_cfg(cfg[:3], pos, cfg[3:], yaw=yaw)

    def _seq_ok(self, seq, world, held, closed):
        tot = 0
        for i in range(len(seq) - 1):
            if i > 0 and not self._config_ok(seq[i], world, None, held, closed):
                return None
            if not self._path_ok(seq[i], seq[i + 1], world, held=held, closed=closed):
                return None
            tot += nsteps(seq[i], seq[i + 1])
        return tot

    def _best_path(self, cur, goal, world, held=None, closed=False):
        """Return (list of waypoints excluding cur, total steps) or None."""
        if self._path_ok(cur, goal, world, held=held, closed=closed):
            return [goal], nsteps(cur, goal)
        best = None
        vias = []
        for dz in (0.1, 0.2):
            l1 = self._lift(cur, dz)
            l2 = self._lift(goal, dz)
            if l1 is not None:
                vias.append([l1])
            if l2 is not None:
                vias.append([l2])
            if l1 is not None and l2 is not None:
                vias.append([l1, l2])
        for v in vias:
            tot = self._seq_ok([cur] + v + [goal], world, held, closed)
            if tot is not None and (best is None or tot < best[1]):
                best = (v + [goal], tot)
        if best is None or (ROUTE_ALL and best[1] > nsteps(cur, goal)):
            r = self._route(cur, goal, world, held, closed)
            if r is not None and (best is None or r[1] < best[1]):
                best = r
        return best

    def _base_graph(self, cur, goal):
        ring = [(-0.72, -1.02), (0.0, -1.02), (0.72, -1.02), (0.72, 0.0),
                (0.72, 1.02), (0.0, 1.02), (-0.72, 1.02), (-0.72, 0.0),
                (-0.66, -0.99), (0.66, -0.99), (0.66, 0.99), (-0.66, 0.99)]
        y0, y1 = float(cur[2]), float(goal[2])
        pts = [tuple(cur[:2]), tuple(goal[:2])] + ring
        for p in (cur, goal):
            if abs(p[1]) < 0.75:
                pts.append((math.copysign(0.66, p[0]), float(p[1])))
                pts.append((math.copysign(0.72, p[0]), float(p[1])))
            if abs(p[0]) < 0.45:
                pts.append((float(p[0]), math.copysign(0.99, p[1])))
                pts.append((float(p[0]), math.copysign(1.02, p[1])))
        yaws = (y0, y1)
        # states: (point index, yaw index)
        S = [(i, j) for i in range(len(pts)) for j in range(2)]
        start = (0, 0); goal_s = (1, 1)

        def ecost(a, b):
            pa, pb = pts[a[0]], pts[b[0]]
            dyaw = abs(wrap(yaws[b[1]] - yaws[a[1]]))
            return max(abs(pb[0] - pa[0]), abs(pb[1] - pa[1]), dyaw) / STEP

        def edge_ok(a, b):
            pa, pb = pts[a[0]], pts[b[0]]
            ya, yb = yaws[a[1]], yaws[b[1]]
            dy = wrap(yb - ya)
            n = int(math.ceil(max(abs(pb[0] - pa[0]), abs(pb[1] - pa[1]), abs(dy)) / STEP - 1e-7))
            for k in range(1, n + 1):
                f = k / n
                if not base_ok((pa[0] + (pb[0] - pa[0]) * f, pa[1] + (pb[1] - pa[1]) * f, ya + dy * f)):
                    return False
            return True
        dist = {s: 1e9 for s in S}; prev = {}; dist[start] = 0.0; done = set()
        while True:
            u = None; du = 1e9
            for s in S:
                if s not in done and dist[s] < du:
                    u, du = s, dist[s]
            if u is None or u == goal_s:
                break
            done.add(u)
            for v in S:
                if v in done or v == u:
                    continue
                c = du + ecost(u, v) + 0.01
                if c < dist[v] and edge_ok(u, v):
                    dist[v] = c; prev[v] = u
        if dist[goal_s] >= 1e9:
            return None
        path = [goal_s]
        while path[-1] != start:
            path.append(prev[path[-1]])
        return pts, yaws, path[::-1], dist[goal_s], ecost

    def _base_cost(self, b0, b1):
        key = tuple(round(float(v), 2) for v in (b0[0], b0[1], wrap(b0[2]), b1[0], b1[1], wrap(b1[2])))
        if key in self.bcache:
            return self.bcache[key]
        n = int(math.ceil(max(abs(b1[0] - b0[0]), abs(b1[1] - b0[1]), abs(wrap(b1[2] - b0[2]))) / STEP - 1e-7))
        ok = True
        for k in range(1, n + 1):
            f = k / n
            if not base_ok((b0[0] + (b1[0] - b0[0]) * f, b0[1] + (b1[1] - b0[1]) * f, b0[2] + wrap(b1[2] - b0[2]) * f)):
                ok = False; break
        if ok:
            c = float(n)
        else:
            gp = grid_path(b0, b1, base_ok) if GRID else None
            if gp is not None:
                c = float(len(gp))
            else:
                r = self._base_graph(np.asarray(b0, float), np.asarray(b1, float))
                c = 60.0 if r is None else r[3]
        self.bcache[key] = c
        return c

    def _lift_route(self, cur, goal, gp, world, held, closed):
        best = None
        po = held[0] if held is not None else None
        for dz in (0.1, 0.2):
            l1 = self._lift(cur, dz)
            l2 = self._lift(goal, dz)
            if l1 is None or l2 is None:
                continue
            d1 = nsteps(cur, np.concatenate([cur[:3], l1[3:]]))
            d2 = nsteps(np.concatenate([cur[:3], l1[3:]]), np.concatenate([cur[:3], l2[3:]]))
            d3 = nsteps(l2, goal)
            k = len(gp)
            T = max(k, d1 + d2 + d3)
            bases = [tuple(cur[:3])] * (T - k) + list(gp)
            a0, a1, a2, a3 = cur[3:], l1[3:], l2[3:], goal[3:]
            def lerp(a, b, f):
                d = cfg_diff(np.concatenate([[0, 0, 0], a]), np.concatenate([[0, 0, 0], b]))[3:]
                r = a + d * f
                return np.where(CONT, wrap(r), r)
            seq = [cur]
            for i in range(1, T):
                if i <= d1:
                    arm = lerp(a0, a1, i / max(d1, 1))
                elif i <= d1 + d2:
                    arm = lerp(a1, a2, (i - d1) / max(d2, 1))
                elif i < T - d3:
                    arm = a2
                else:
                    arm = lerp(a2, a3, (i - (T - d3)) / max(d3, 1))
                c = np.concatenate([np.asarray(bases[i - 1], float), arm])
                seq.append(c)
            seq.append(goal)
            tot = self._seq_ok(seq, world, held, closed)
            if tot is not None and (best is None or tot < best[1]):
                best = (seq[1:], tot)
            if best is not None and best[1] <= k:
                break
        return best

    def _route(self, cur, goal, world, held, closed):
        """Route the base around the table through ring waypoints (yaw-aware)."""
        best = None
        gp = grid_path(cur[:3], goal[:3], base_ok) if GRID else None
        if gp is not None:
            dq = cfg_diff(cur, goal)
            k = len(gp)
            for mode in (0, 1, 2):
                seq = [cur]
                arm = cur[3:].copy()
                for i, b in enumerate(gp[:-1], start=1):
                    c = cur + dq * (i / k)
                    c[0], c[1], c[2] = b
                    if mode == 1:
                        c[3:] = Q0
                    elif mode == 2:
                        rem = k - i
                        b3 = cur[:3]
                        dd = cfg_diff(np.concatenate([b3, arm]), np.concatenate([b3, Q0]))[3:]
                        cand = arm + np.clip(dd, -STEP, STEP)
                        cand = np.where(CONT, wrap(cand), cand)
                        dg = cfg_diff(np.concatenate([b3, cand]), goal)[3:]
                        if np.abs(dg).max() > STEP * rem * 0.999:
                            dg = cfg_diff(np.concatenate([b3, arm]), goal)[3:]
                            cand = arm + np.clip(dg, -STEP, STEP)
                            cand = np.where(CONT, wrap(cand), cand)
                        arm = cand
                        c[3:] = arm
                    seq.append(c)
                seq.append(goal)
                tot = self._seq_ok(seq, world, held, closed)
                if tot is not None and (best is None or tot < best[1]):
                    best = (seq[1:], tot)
            if LIFT_ROUTE and (best is None or best[1] > k):
                r3 = self._lift_route(cur, goal, gp, world, held, closed)
                if r3 is not None and (best is None or r3[1] < best[1]):
                    best = r3
            if best is not None and best[1] <= k:
                return best
        res = self._base_graph(cur, goal)
        if res is None:
            return best
        pts, yaws, path, dist_goal, ecost = res
        start = (0, 0); goal_s = (1, 1)
        total = max(dist_goal, 1e-6)
        dq = cfg_diff(cur, goal)
        for mode in (0, 1):
            seq = [cur]
            acc = 0.0
            for i in range(1, len(path) - 1):
                acc += ecost(path[i - 1], path[i]) + 0.01
                f = acc / total
                c = cur + dq * f
                c[0], c[1] = pts[path[i][0]]
                c[2] = yaws[path[i][1]]
                if mode == 1:
                    c[3:] = Q0
                seq.append(c)
            seq.append(goal)
            tot = self._seq_ok(seq, world, held, closed)
            if tot is not None and (best is None or tot < best[1]):
                best = (seq[1:], tot)
        return best

    # ------------------------------------------------------------ options
    def _base_list(self, cur, x, y, extra):
        bases = [tuple(cur[:3])] + BASE_CANDS + block_bases(x, y, dense=extra)
        cand = []; seen = set()
        for base in bases:
            key = (round(base[0], 3), round(base[1], 3), round(wrap(base[2]), 3))
            if key in seen:
                continue
            seen.add(key)
            if not base_ok(base):
                continue
            sh = shoulder_xy(base)
            dist = math.hypot(x - sh[0], y - sh[1])
            if dist > 0.86 or dist < 0.2:
                continue
            lb = int(math.ceil(max(abs(base[0] - cur[0]), abs(base[1] - cur[1]),
                                   abs(wrap(base[2] - cur[2]))) / STEP - 1e-7))
            cand.append((lb, base))
        if SHORTEN:
            extra_c = []
            for lb, base in cand:
                if lb < 2:
                    continue
                d = np.array([base[0] - cur[0], base[1] - cur[1], wrap(base[2] - cur[2])])
                m = np.abs(d).max()
                for kk in (lb - 1,):
                    sc = kk * STEP / m * 0.999
                    nb = (cur[0] + d[0] * sc, cur[1] + d[1] * sc, wrap(cur[2] + d[2] * sc))
                    if not base_ok(nb):
                        continue
                    sh = shoulder_xy(nb)
                    if math.hypot(x - sh[0], y - sh[1]) > 0.86:
                        continue
                    extra_c.append((kk, nb))
            cand += extra_c
        cand.sort(key=lambda t: t[0])
        return cand

    def _grasp_options(self, cur, name, b, world, extra=False, max_ik=None):
        x, y, z, yaw = b
        pos = np.array([x, y, z + GRASP_DZ])
        out = []; best = None; n_ik = 0
        max_ik = max_ik or (80 if extra else 36)
        for lb, base in self._base_list(cur, x, y, extra):
            if best is not None and lb > best + 1 and len(out) >= 4:
                break
            if n_ik >= max_ik:
                break
            same = np.allclose(base, cur[:3])
            inits = [cur[3:], Q0]
            for qi in inits:
                n_ik += 1
                c = self._ik_cfg(base, pos, qi, yaw=yaw, multi=(qi is Q0))
                if c is None:
                    continue
                vs = []
                for k in range(4):
                    cc = c.copy(); cc[9] = wrap(cc[9] + k * math.pi / 2)
                    vs.append(cc)
                if REFINE:
                    vs.sort(key=lambda v: nsteps(cur, v))
                    v0 = vs[0]
                    if nsteps(cur, v0) > lb:
                        R = fk_full(v0[:3], v0[3:])[4]
                        qr = ik_refine(v0[:3], pos, v0[3:], cur[3:], yaw=math.atan2(R[1, 1], R[0, 1]))
                        if qr is not v0[3:]:
                            vs.insert(0, np.concatenate([v0[:3], qr]))
                for cc in vs:
                    if ROLL_TOL > 0:
                        d = cfg_diff(cur, cc)
                        m_other = max(np.abs(d[:9]).max(), lb * STEP)
                        ex = abs(d[9]) - m_other
                        if ex > 0:
                            cc = cc.copy()
                            cc[9] = wrap(cc[9] - math.copysign(min(ex, ROLL_TOL), d[9]))
                    if not self._config_ok(cc, world, exclude=name, closed=False, margin=0.0):
                        continue
                    sc = max(nsteps(cur, cc), lb)
                    out.append((sc, cc))
                    best = sc if best is None else min(best, sc)
        out.sort(key=lambda t: t[0])
        return out

    def _slot_candidates(self, placed, yaw=0.0):
        px, py = self.plate_c; hx, hy = self.plate_h
        r = 0.035 * (abs(math.cos(yaw)) + abs(math.sin(yaw)))
        lim = hx - r - PLATE_MARGIN
        g = np.linspace(-lim, lim, 9)
        out = []
        for sx in g:
            for sy in g:
                c = (px + sx, py + sy)
                ok = True
                for pb in placed:
                    if rect_overlap_2d(c, yaw, (0.035, 0.035), pb[:2], pb[3], (0.035, 0.035), margin=0.004):
                        ok = False; break
                if ok:
                    out.append(c)
        return out

    def _room_left(self, placed, n):
        if n <= 0:
            return True
        cands = self._slot_candidates(placed)
        if not cands:
            return False
        if n == 1:
            return True
        for c in cands:
            if self._room_left(placed + [(c[0], c[1], 0.781, 0.0)], n - 1):
                return True
        return False

    def _place_options(self, gcfg, world, held, placed, nrem, extra=False, max_ik=None, nslots=4):
        """Place configs from config gcfg (holding). Returns sorted list of (steps, cfg)."""
        po, Rrel = held
        self._placed_ctx = placed
        _, _, _, tool, Rg0 = fk_full(gcfg[:3], gcfg[3:])
        Rb = Rg0 @ Rrel
        byaw = math.atan2(Rb[1, 0], Rb[0, 0])
        good = []
        for fy in (0.0, byaw):
            slots = self._slot_candidates(placed, fy)
            slots.sort(key=lambda c: (c[0] - tool[0]) ** 2 + (c[1] - tool[1]) ** 2)
            ng = 0
            for c in slots:
                if nrem > 0 and not self._room_left(placed + [(c[0], c[1], 0.781, fy)], nrem):
                    continue
                good.append((c[0], c[1], None if fy == 0.0 else fy))
                ng += 1
                if ng >= nslots:
                    break
        out = []
        max_ik = max_ik or (40 if extra else 12)
        n_ik = 0
        for c in good:
            best = None
            for lb, base in self._base_list(gcfg, c[0], c[1], extra):
                if best is not None and lb > best:
                    break
                if n_ik >= max_ik * (1 + good.index(c)):
                    break
                for high in (True, False):
                    n_ik += 1
                    res = self._place_ik(base, gcfg, c, po, Rrel, high, world, held)
                    if res is None:
                        continue
                    s = max(nsteps(gcfg, res), lb)
                    out.append((s, res))
                    best = s if best is None else min(best, s)
                    break
        out.sort(key=lambda t: t[0])
        return out

    def _footprint_ok(self, cfg, po, Rrel):
        _, _, _, tool, R = fk_full(cfg[:3], cfg[3:])
        cen = tool + R @ po
        Rb = R @ Rrel
        yaw = math.atan2(Rb[1, 0], Rb[0, 0])
        r = 0.035 * (abs(math.cos(yaw)) + abs(math.sin(yaw)))
        px, py = self.plate_c; hx, hy = self.plate_h
        if abs(cen[0] - px) + r > hx - 0.002 or abs(cen[1] - py) + r > hy - 0.002:
            return False
        for pb in self._placed_ctx:
            if rect_overlap_2d(cen[:2], yaw, (0.035, 0.035), pb[:2], pb[3], (0.035, 0.035), margin=0.002):
                return False
        return True

    def _place_ik(self, base, gcfg, c, po, Rrel, high, world, held):
        _, _, _, _, Rg = fk_full(gcfg[:3], gcfg[3:])
        Rb = Rg @ Rrel
        byaw = math.atan2(Rb[1, 0], Rb[0, 0])
        tyaw = math.atan2(Rg[1, 1], Rg[0, 1])
        off = tyaw - byaw
        zlow = TABLE_TOP + BLOCK_HZ + 0.008
        if high:
            cz = float(fk_full(gcfg[:3], gcfg[3:])[3][2]) - po[0]
            zc = min(max(cz, zlow), zlow + 0.25)
        else:
            zc = zlow
        if c[2] is None:
            k0 = round(wrap(tyaw - off) / (math.pi / 2))
            psi = wrap(k0 * math.pi / 2 + off)
        else:
            psi = tyaw
        sols = []
        for qi in (gcfg[3:], Q0):
            q, ok = ik_down(base, np.array([c[0], c[1], zc]), qi, yaw=psi, p_off=po)
            if not ok:
                q, ok = ik_down(base, np.array([c[0], c[1], zc]), q, yaw=psi, p_off=po, iters=170)
            if ok:
                sols.append(q)
                break
        cands = []
        for q in sols:
            for k in range(4):
                qq = q.copy(); qq[6] = wrap(qq[6] + k * math.pi / 2)
                cfg = np.concatenate([np.asarray(base, float), qq])
                cands.append((nsteps(gcfg, cfg), k, cfg))
        cands.sort(key=lambda t: (t[0], t[1]))
        if REFINE and cands:
            sc0, k0_, c0_ = cands[0]
            lbb = int(math.ceil(np.abs(cfg_diff(gcfg, c0_))[:3].max() / STEP - 1e-7))
            if sc0 > lbb:
                R = fk_full(c0_[:3], c0_[3:])[4]
                qr = ik_refine(c0_[:3], np.array([c[0], c[1], zc]), c0_[3:], gcfg[3:],
                               yaw=math.atan2(R[1, 1], R[0, 1]), p_off=po)
                if qr is not c0_[3:]:
                    cr = np.concatenate([c0_[:3], qr])
                    cands.insert(0, (nsteps(gcfg, cr), k0_, cr))
        best = None
        for sc, k, cfg in cands:
            if k and not self._footprint_ok(cfg, po, Rrel):
                continue
            if not self._config_ok(cfg, world, None, held, closed=True, margin=0.0):
                continue
            best = (sc, cfg)
            break
        return None if best is None else best[1]

    # ------------------------------------------------------------ top level
    def _home(self, cfg):
        return np.concatenate([[-0.75, 0.0, 0.0], Q0])

    def _make_plan(self, cfg, blocks, holding, held, gtf, gq):
        extra = self.fail >= 2 or self.no_plan > 0
        cheap = self.t_used > 25.0
        placed = [b for n, b in blocks.items() if n != held and self._on_plate(b)]
        remaining = [n for n, b in blocks.items() if n != held and not self._on_plate(b)]
        if holding and held is not None:
            heldinfo = (gtf, quat_to_R(gq))
            w2 = World({n: b for n, b in blocks.items() if n != held})
            opts = self._place_options(cfg, w2, heldinfo, placed, len(remaining), extra=extra)
            futc = []
            if remaining and not cheap:
                for n2 in remaining:
                    for s2, gc2 in self._grasp_options(cfg, n2, blocks[n2], w2, max_ik=10)[:3]:
                        futc.append(gc2)
            bestp = None; bestv = None
            for s, pc in opts[:8]:
                if bestv is not None and s >= bestv and not futc:
                    break
                bp = self._best_path(cfg, pc, w2, held=heldinfo, closed=True)
                if bp is None:
                    continue
                v = bp[1]
                if futc:
                    v += FUT_W * min(max(self._base_cost(pc[:3], g[:3]), nsteps(pc, g)) for g in futc)
                if bestv is None or v < bestv:
                    bestp, bestv = bp, v
            if bestp is not None:
                wps = bestp[0]
                return [(w, 0.0) for w in wps[:-1]] + [(wps[-1], 1.0)]
            return self._fallback(cfg, w2, heldinfo, True)
        if not remaining:
            return []
        world = World(blocks)
        cands = []
        for n in remaining:
            for s, gc in self._grasp_options(cfg, n, blocks[n], world, extra=extra)[:4]:
                cands.append((s, n, gc))
        cands.sort(key=lambda t: t[0])
        # lookahead: add estimated place cost
        scored = []
        for s, n, gc in cands[:(4 if cheap else 10)]:
            bp = self._best_path(cfg, gc, world, closed=False)
            if bp is None:
                continue
            wps, tot = bp
            pc = 0
            if not cheap:
                b = blocks[n]
                # held offset estimate: block centre GRASP_DZ below tool, aligned
                po = np.array([GRASP_DZ, 0.0, 0.0])
                _, _, _, _, Rg = fk_full(gc[:3], gc[3:])
                Rrel = Rg.T @ Rz(b[3])
                po_opts = self._place_options(gc, World({k: v for k, v in blocks.items() if k != n}),
                                              (po, Rrel), placed, len(remaining) - 1, max_ik=4, nslots=1)
                pc = po_opts[0][0] if po_opts else 30
                if po_opts and len(remaining) > 1:
                    pcfg = po_opts[0][1]
                    fut = 1e9
                    for s2, n2, gc2 in cands:
                        if n2 == n:
                            continue
                        bc = self._base_cost(pcfg[:3], gc2[:3])
                        fut = min(fut, max(bc, nsteps(pcfg, gc2)))
                    if fut < 1e9:
                        pc += FUT_W * fut
            scored.append((tot + pc, tot, n, wps))
        if not scored:
            return self._fallback(cfg, world, None, False)
        scored.sort(key=lambda t: (t[0], t[1]))
        if getattr(self, 'debug', False):
            for t in scored:
                print('  scored', round(t[0], 1), t[1], t[2], np.round(t[3][-1][:3], 2))
        _, tot, n, wps = scored[0]
        self.target = n
        return [(w, 0.0) for w in wps[:-1]] + [(wps[-1], -1.0)]

    def _fallback(self, cfg, world, held, closed):
        self.no_plan += 1
        home = self._home(cfg)
        if nsteps(cfg, home) > 0:
            bp = self._best_path(cfg, home, world, held=held, closed=closed)
            if bp is not None:
                return [(w, 0.0) for w in bp[0]]
        # random small perturbation of the arm
        rng = np.random.default_rng(self.no_plan)
        c = cfg.copy(); c[3:] += rng.uniform(-0.2, 0.2, 7)
        c[3:] = np.where(CONT, wrap(c[3:]), np.clip(c[3:], LO, HI))
        return [(c, 0.0)]

    def reset(self, state, info):
        self.plan = []
        self.last_cfg = None
        self.last_cmd = None
        self.fail = 0
        self.target = None
        self.bad_cfgs = []
        self.bad_bases = []
        self.margin = 0.01
        self.no_plan = 0
        self.t_used = float(os.environ.get("PR2_T0", "0"))
        self.rng = np.random.default_rng(0)
        self.bcache = {}

    def get_action(self, state):
        try:
            return self._get_action(state)
        except Exception:
            self.plan = []
            self.last_cmd = None
            self.no_plan += 1
            rng = np.random.default_rng(self.no_plan)
            a = np.zeros(11, dtype=np.float32)
            a[3:10] = rng.uniform(-0.1, 0.1, 7)
            return a

    def _get_action(self, state):
        cfg, blocks, holding, held, gtf, gq = self._parse(state)
        # detect rejection / failed grasp
        if self.last_cmd is not None:
            moved = np.abs(cfg_diff(self.last_cfg, cfg)).max() > 1e-6
            wanted = np.abs(self.last_cmd[:10]).max() > 1e-6
            if wanted and not moved:
                self.plan = []
                self.fail += 1
                bc = self.last_cfg + self.last_cmd[:10]
                self.bad_cfgs.append(bc)
                if np.abs(self.last_cmd[:3]).max() > 1e-6:
                    try:
                        if self._config_ok(bc, World({n: b for n, b in blocks.items() if n != held}),
                                           None, None, holding, margin=0.0):
                            self.bad_bases.append(bc[:3].copy())
                    except Exception:
                        pass
                self.margin = min(0.06, self.margin + 0.015)
            elif self.last_cmd[10] < -0.5 and not holding:
                self.plan = []
                self.fail += 1
            elif self.last_cmd[10] > 0.5 and holding:
                self.plan = []
                self.fail += 1
        if self.plan:
            # drop reached waypoints
            while self.plan and nsteps(cfg, self.plan[0][0]) == 0 and self.plan[0][1] == 0.0:
                self.plan.pop(0)
        if self.plan and self.plan[0][1] != 0.0 and nsteps(cfg, self.plan[0][0]) == 0:
            # arrived but gripper not executed? execute now
            pass
        if not self.plan:
            t0 = time.time()
            self.plan = self._make_plan(cfg, blocks, holding, held, gtf, gq)
            self.t_used += time.time() - t0
        act = np.zeros(11, dtype=np.float32)
        if not self.plan:
            self.last_cfg = cfg; self.last_cmd = act
            return act
        tgt, grip = self.plan[0]
        n = nsteps(cfg, tgt)
        d = cfg_diff(cfg, tgt)
        if n >= 1:
            d = d / n
        act[:10] = np.clip(d, -STEP, STEP)
        if n <= 1:
            act[10] = grip
            self.plan.pop(0)
        self.last_cfg = cfg
        self.last_cmd = act.copy()
        return act
