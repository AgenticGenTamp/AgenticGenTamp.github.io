"""Obstruction3D approach: clear obstructions off the target region, then place
the target block on the region.  Uses a calibrated Kinova Gen3 FK/IK model
(see kin.py) and a Cartesian waypoint controller."""
import numpy as np
from kin import fk, ik, down_R, jac, rot_err

TABLE_Z = 0.075
TABLE_X = (0.10, 0.50)
TABLE_Y = (-0.40, 0.40)
ROUTE_MARGIN = 0.02
SAFE_MARGIN = 0.03
VSTEP = 0.08
HSTEP = 0.10
GRASP_DZ = 0.031       # tool height above object top when closing
PLACE_EPS = 0.0015     # clearance above support surface when releasing
MAX_DQ = 0.2
JLIM = {1: 2.41, 3: 2.66, 5: 2.23}   # limited joints (0-indexed): 2,4,6 in 1-index


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space

    # ------------------------------------------------------------------ utils
    def _read(self, s):
        self.s = s
        self.R = s.get_object_from_name('robot')
        self.q = np.array([s.get(self.R, f'joint_{i}') for i in range(1, 8)])
        self.base = tuple(s.get(self.R, f) for f in ('pos_base_x', 'pos_base_y', 'pos_base_rot'))
        self.grasped = s.get(self.R, 'grasp_active') > 0.5
        self.objs = {}
        for n in s.get_object_names():
            if n == 'robot':
                continue
            o = s.get_object_from_name(n)
            try:
                p = np.array([s.get(o, f) for f in ('pose_x', 'pose_y', 'pose_z')])
                he = np.array([s.get(o, 'half_extent_' + c) for c in 'xyz'])
            except Exception:
                continue
            self.objs[n] = (p, he)

    def _obstructions(self):
        return sorted(n for n in self.objs if n not in ('target_block', 'target_region'))

    def reset(self, state, info):
        self._read(state)
        self.phase = 'plan'
        self.task = None
        self.path = []
        self.last_q = None
        self.last_cmd = None
        self.blocked = 0
        self.scale = 1.0
        self.last_final = False
        self.tilt = (0.0, 0.0)
        self.fail = {}
        self.t = 0
        self.wait = 0

    # ------------------------------------------------------------ geometry
    def _final_block_xy(self):
        p, _ = self.objs['target_region']
        return p[:2].copy()

    def _needs_clearing(self, n):
        """Obstruction overlaps the target block's final footprint (+margin)
        or would block the gripper when placing the block."""
        rp, rhe = self.objs['target_region']
        bp, bhe = self.objs['target_block']
        p, he = self.objs[n]
        c = self._final_block_xy()
        m = 0.008
        ox = abs(p[0] - c[0]) < he[0] + bhe[0] + m
        oy = abs(p[1] - c[1]) < he[1] + bhe[1] + m
        rx = abs(p[0] - rp[0]) < he[0] + rhe[0] + 0.003
        ry = abs(p[1] - rp[1]) < he[1] + rhe[1] + 0.003
        if (ox and oy) or (rx and ry and p[2] - he[2] > rp[2] + rhe[2] - 0.003):
            return True
        return False

    # finger model (tool frame): two boxes centred at +-FX, half extents
    FX, FHX, FHY, TIP = 0.061, 0.025, 0.017, 0.032

    def _rect_hit(self, cxy, ax, ay, hx, hy, p, he, m):
        d = p[:2] - cxy
        for axis, r1 in ((np.array([1.0, 0]), abs(ax[0]) * hx + abs(ay[0]) * hy),
                         (np.array([0, 1.0]), abs(ax[1]) * hx + abs(ay[1]) * hy)):
            if abs(d @ axis) > r1 + (he[0] if axis[0] else he[1]) + m:
                return False
        for axis, r1 in ((ax, hx), (ay, hy)):
            r2 = abs(axis[0]) * he[0] + abs(axis[1]) * he[1]
            if abs(d @ axis) > r1 + r2 + m:
                return False
        return True

    def _finger_hits(self, name, xy, yaw, tool_z, held=None, m=0.004):
        """Number of objects hit by the finger boxes (tips at tool_z - TIP)."""
        c, s_ = np.cos(yaw), np.sin(yaw)
        ax = np.array([c, s_]); ay = np.array([-s_, c])
        tip = tool_z - self.TIP
        hits = 0
        for n, (p, he) in self.objs.items():
            if n in (name, held, 'target_region'):
                continue
            if p[2] + he[2] < tip - 0.003:
                continue
            for sgn in (1, -1):
                if self._rect_hit(xy + sgn * self.FX * ax, ax, ay, self.FHX, self.FHY, p, he, m):
                    hits += 1
                    break
        return hits

    def _self_overlap(self, he, yaw):
        """Extent of object along finger axis beyond the finger gap."""
        c, s_ = abs(np.cos(yaw)), abs(np.sin(yaw))
        ext = c * he[0] + s_ * he[1]
        return max(0.0, ext - 0.036)

    def _cur_yaw(self):
        M = fk(self.base, self.q)
        return np.arctan2(M[1, 0], M[0, 0])

    def _yaw_options(self, name, xy, tool_z, cur, held=None):
        p, he = self.objs[name]
        out = []
        for k in range(12):
            y = cur + k * np.pi / 12
            if k > 6:
                y -= np.pi
            dev = abs(y - cur)
            hits = self._finger_hits(name, xy, y, tool_z, held)
            ov = self._self_overlap(he, y)
            out.append((hits * 10 + ov * 50 + 0.2 * dev, hits, y))
        out.sort()
        return out

    def _free_spot(self, name, yaw):
        """Pick a table location for obstruction `name` (held at yaw)."""
        p0, he0 = self.objs[name]
        c, s_ = abs(np.cos(yaw)), abs(np.sin(yaw))
        ex = np.array([c * he0[0] + s_ * he0[1], s_ * he0[0] + c * he0[1]])
        rp, rhe = self.objs['target_region']
        bp, bhe = self.objs['target_block']
        tool_z = TABLE_Z + 2 * he0[2] + GRASP_DZ
        best, bestv = None, -1e9
        for x in np.linspace(TABLE_X[0] + 0.05, TABLE_X[1] - 0.05, 15):
            for y in np.linspace(TABLE_Y[0] + 0.05, TABLE_Y[1] - 0.05, 25):
                xy = np.array([x, y])
                clear = 1.0
                for n, (p, he) in self.objs.items():
                    if n == name:
                        continue
                    ext = he[:2].copy()
                    if n == 'target_region':
                        ext = ext + 0.02
                    dx = max(0.0, abs(x - p[0]) - ext[0] - ex[0])
                    dy = max(0.0, abs(y - p[1]) - ext[1] - ex[1])
                    clear = min(clear, np.hypot(dx, dy))
                if clear < 0.012:
                    continue
                # keep away from the target block (future pick) and region
                kb = min(np.hypot(max(0, abs(x - bp[0]) - bhe[0] - ex[0]), max(0, abs(y - bp[1]) - bhe[1] - ex[1])),
                         np.hypot(max(0, abs(x - rp[0]) - rhe[0] - ex[0]), max(0, abs(y - rp[1]) - rhe[1] - ex[1])))
                if self._finger_hits(name, xy, yaw, tool_z) > 0:
                    continue
                bad = any(np.hypot(x - b[0], y - b[1]) < 0.04 for b in self.fail.get(name, []))
                if bad:
                    continue
                v = min(clear, 0.05) + min(kb, 0.10) - 0.08 * np.hypot(x - p0[0], y - p0[1]) - 0.2 * self._tilt_needed(np.hypot(*(xy - self._mount())))
                if v > bestv:
                    bestv, best = v, xy
        if best is None:
            best = np.array([0.3, 0.3 if p0[1] < 0 else -0.3])
        return best

    # ------------------------------------------------------------- planning
    def _pick_choice(self, n, cur):
        p, he = self.objs[n]
        tz = p[2] + he[2] + GRASP_DZ
        opts = self._yaw_options(n, p[:2], tz, cur)
        att = self.fail.get(('grasp', n), 0)
        return opts[min(att, len(opts) - 1)]

    def _plan_next(self):
        cur = self._cur_yaw()
        tool = fk(self.base, self.q)[:3, 3]
        need = [n for n in self._obstructions() if self._needs_clearing(n)]
        name = None
        if need:
            scored = []
            for n in need:
                v, hits, y = self._pick_choice(n, cur)
                d = np.hypot(*(self.objs[n][0][:2] - tool[:2]))
                scored.append((hits, self.fail.get(('grasp', n), 0), d, n, y))
            scored.sort()
            if scored[0][0] > 0:
                # try to move a blocker first
                others = [n for n in self._obstructions() if n not in need] + ['target_block']
                alt = []
                for n in others:
                    v, hits, y = self._pick_choice(n, cur)
                    if hits == 0:
                        d = np.hypot(*(self.objs[n][0][:2] - tool[:2]))
                        alt.append((self.fail.get(('grasp', n), 0), d, n, y))
                # only blockers that are actually hit by best option of first needed
                n0, y0 = scored[0][3], scored[0][4]
                p0, he0 = self.objs[n0]
                blockers = []
                for f, d, n, y in sorted(alt):
                    saved = self.objs.pop(n)
                    h = self._finger_hits(n0, p0[:2], y0, p0[2] + he0[2] + GRASP_DZ)
                    self.objs[n] = saved
                    if h < scored[0][0]:
                        blockers.append((n, y))
                if blockers:
                    name, yaw = blockers[0]
            if name is None:
                name, yaw = scored[0][3], scored[0][4]
            p, he = self.objs[name]
            self.task = dict(name=name, dest=self._free_spot(name, yaw), surf=TABLE_Z)
        else:
            rp, rhe = self.objs['target_region']
            name = 'target_block'
            v, hits, yaw = self._pick_choice(name, cur)
            self.task = dict(name=name, dest=self._final_block_xy(), surf=rp[2] + rhe[2])
        n = name
        p, he = self.objs[n]
        self.task['yaw'] = yaw
        self.task['yaw_place'] = yaw
        self._set_tilt(p[:2], self.task['dest'])
        self.phase = 'to_pick'
        tool = fk(self.base, self.q)[:3, 3]
        pick = np.array([p[0], p[1], p[2] + he[2] + GRASP_DZ])
        path = self._route(tool, pick, None, 0.036, 0.09)
        self.path = self._mk(path, self.task['yaw'])
        return
        self._set_path(path, self.task['yaw'])

    def _mount(self):
        return np.array([self.base[0] + 0.12 * np.cos(self.base[2]),
                         self.base[1] + 0.12 * np.sin(self.base[2])])

    @staticmethod
    def _tilt_needed(r):
        return float(np.clip((0.15 - r) * 8.0, 0.0, 0.35))

    def _set_tilt(self, a, b):
        m = self._mount()
        best = (0.0, 0.0)
        for xy in (a, b):
            d = np.asarray(xy[:2]) - m
            t = self._tilt_needed(np.hypot(*d))
            if t > best[0]:
                best = (t, np.arctan2(d[1], d[0]))
        self.tilt = best

    def _Rot(self, yaw):
        R = down_R(yaw)
        t, phi = self.tilt
        if t <= 0:
            return R
        ax = np.array([-np.sin(phi), np.cos(phi), 0.0])
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return (np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K) @ R

    def _yaw_for(self, xy):
        mx = self.base[0] + 0.12 * np.cos(self.base[2])
        my = self.base[1] + 0.12 * np.sin(self.base[2])
        return np.arctan2(xy[1] - my, xy[0] - mx)

    @staticmethod
    def _match_yaw(y0, ydes):
        # object must keep orientation up to pi
        k = np.round((ydes - y0) / np.pi)
        return y0 + k * np.pi

    def _need_z(self, xy, held, depth, rad, skip=()):
        """Min tool z at xy so nothing within rad (xy) is hit."""
        need = TABLE_Z + depth + 0.01
        for n, (p, he) in self.objs.items():
            if n == held or n in skip:
                continue
            dx = max(0.0, abs(xy[0] - p[0]) - he[0])
            dy = max(0.0, abs(xy[1] - p[1]) - he[1])
            if dx * dx + dy * dy < rad * rad:
                need = max(need, p[2] + he[2] + depth + ROUTE_MARGIN)
        return need

    def _mk(self, pts, yaw):
        return [(p, yaw, 0.006 if any(np.linalg.norm(p - t) < 1e-6 for t in self.tight) else 0.03) for p in pts]

    def _route(self, start, goal, held, depth, rad, approach=0.035, skip_start=(), skip_goal=()):
        """Waypoints from start to goal: vertical exit, diagonal/flat transit, vertical entry."""
        a = np.array([start[0], start[1], max(start[2], min(start[2] + approach, 0.30))])
        g = np.array([goal[0], goal[1], goal[2] + approach])
        # vertical exit only needed if tool low near objects
        d = np.linalg.norm(g[:2] - a[:2])
        ns = max(2, int(np.ceil(d / 0.01)))
        need = []
        for i in range(ns + 1):
            f = i / ns
            xy = a[:2] + (g[:2] - a[:2]) * f
            sk = set()
            if f < 0.3:
                sk |= set(skip_start)
            if f > 0.7:
                sk |= set(skip_goal)
            need.append(self._need_z(xy, held, depth, rad, sk))
        need = np.array(need)
        zs = a[2] + (g[2] - a[2]) * np.linspace(0, 1, ns + 1)
        pts = []
        self.tight = []
        if start[2] > self._need_z(start[:2], held, depth, rad) + 0.005:
            a = np.asarray(start, float).copy()
        if start[2] < a[2] - 1e-4:
            pts.append(a)
            self.tight.append(a)
        if np.all(zs >= need):
            pts += self._line(a, g, HSTEP)
        else:
            h = max(need.max(), a[2], g[2])
            b = np.array([a[0], a[1], h]); c = np.array([g[0], g[1], h])
            # try to skip vertical parts by diagonals where clear
            pts += self._line(a, b, VSTEP) if h > a[2] + 1e-4 else []
            self.tight.append(b)
            pts += self._line(b, c, HSTEP)
            self.tight.append(c)
            if h > g[2] + 1e-4:
                pts += self._line(c, g, VSTEP)
        pts.append(np.asarray(goal, float))
        return pts

    def _safe_z(self, held, depth):
        top = TABLE_Z
        for n, (p, he) in self.objs.items():
            if n != held:
                top = max(top, p[2] + he[2])
        return max(top + depth + SAFE_MARGIN, 0.18)

    @staticmethod
    def _line(a, b, step):
        d = np.linalg.norm(b - a)
        n = max(1, int(np.ceil(d / step)))
        return [a + (b - a) * (i / n) for i in range(1, n + 1)]

    def _set_path(self, pts, yaw, yaw_end=None):
        if yaw_end is None:
            yaw_end = yaw
        n = len(pts)
        self.path = [(p, yaw + (yaw_end - yaw) * (i + 1) / n) for i, p in enumerate(pts)]

    # -------------------------------------------------------------- control
    def _ik(self, pos, yaw):
        q, err = ik(self.base, self.q, pos, self._Rot(yaw), iters=60)
        for i, lim in JLIM.items():
            q[i] = np.clip(q[i], -lim, lim)
        return q, err

    def _track(self):
        """Return joint delta toward current path, or None when path done."""
        self.last_final = False
        M = fk(self.base, self.q)
        cur = M[:3, 3]
        while self.path:
            pos, yaw = self.path[0][:2]
            final = len(self.path) == 1
            tol = 5e-4 if final else 0.03
            if len(self.path[0]) > 2 and not final:
                tol = self.path[0][2]
            e = np.linalg.norm(cur - pos)
            eo = np.linalg.norm(rot_err(M[:3, :3], self._Rot(yaw)))
            if e < tol and (eo < (0.01 if final else 0.1)):
                self.path.pop(0)
                continue
            break
        if not self.path:
            return None
        # polyline up to first tight corner / final waypoint
        pts = [cur]
        k = 0
        for i, w in enumerate(self.path):
            pts.append(np.asarray(w[0], float))
            k = i
            if (len(w) > 2 and w[2] < 0.01) or i == len(self.path) - 1:
                break
        stop_final = (k == len(self.path) - 1)
        yaw = self.path[k][1]
        segs = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
        cum = np.concatenate([[0.0], np.cumsum(segs)])
        L = cum[-1]

        def at(sv):
            j = int(np.searchsorted(cum, sv, side='right') - 1)
            j = min(max(j, 0), len(segs) - 1)
            f = 0.0 if segs[j] < 1e-9 else (sv - cum[j]) / segs[j]
            return pts[j] + (pts[j + 1] - pts[j]) * min(max(f, 0.0), 1.0)

        s_max = L
        if self.scale < 1.0:
            s_max = min(L, 0.04 * self.scale)
        qt, err = self._ik(at(s_max), yaw)
        m = np.max(np.abs(qt - self.q))
        s_sel = s_max
        if m > MAX_DQ and s_max > 0.005:
            lo, hi = 0.0, s_max
            best = None
            for _ in range(6):
                mid = 0.5 * (lo + hi)
                q2, e2 = self._ik(at(mid), yaw)
                if np.max(np.abs(q2 - self.q)) <= MAX_DQ:
                    lo, best = mid, (q2, e2)
                else:
                    hi = mid
            if best is not None:
                qt, err = best
                s_sel = lo
        dq = qt - self.q
        m = np.max(np.abs(dq))
        self.last_final = bool(stop_final and s_sel >= L - 1e-9 and m <= MAX_DQ
                               and err < 1e-3 and self.scale >= 1.0)
        if m > MAX_DQ:
            dq = dq * (MAX_DQ / m)
        # drop intermediate waypoints we pass this step
        npop = 0
        for i in range(1, len(cum) - 1):
            if cum[i] <= s_sel - 1e-9:
                npop += 1
        if npop and not (m > MAX_DQ):
            del self.path[:npop]
        return dq

    def _act(self, dq=None, grip=0.0):
        a = np.zeros(11, dtype=np.float32)
        if dq is not None:
            a[3:10] = dq
        a[10] = grip
        self.last_q = self.q.copy()
        self.last_cmd = a.copy()
        return a

    def get_action(self, state):
        self._read(state)
        self.t += 1
        # detect blocked motion
        blocked = False
        if self.last_cmd is not None and np.max(np.abs(self.last_cmd[3:10])) > 1e-6:
            if np.allclose(self.q, self.last_q, atol=1e-7):
                blocked = True
        if self.last_cmd is not None and not blocked:
            self.scale = min(1.0, self.scale * 2)
        if blocked and self.scale > 1.0 / 16:
            self.scale *= 0.5
            blocked = False
        elif blocked:
            self.scale = 1.0
        self.last_cmd = None

        for _ in range(8):
            a = self._step_fsm(blocked)
            if a is not None:
                return a
            blocked = False
        return self._act()

    def _step_fsm(self, blocked):
        ph = self.phase
        if ph == 'plan':
            if self.grasped:
                # holding something unexpectedly: lower & release
                self.phase = 'release'
                return None
            self._plan_next()
            return None
        n = self.task['name']
        if ph == 'to_pick':
            if blocked:
                # blocked while approaching: close anyway if near, else lift
                self.phase = 'grasp'
                return None
            dq = self._track()
            if dq is None:
                self.phase = 'grasp'
                return None
            if self.last_final:
                self.phase = 'grasp_c'
                self.path = []
                return self._act(dq, grip=-1.0)
            return self._act(dq)
        if ph == 'grasp_c':
            self.phase = 'grasp'
            self.wait = 0
            return None
        if ph == 'grasp':
            if self.grasped:
                self.phase = 'to_place'
                p, he = self.objs[n]
                tool = fk(self.base, self.q)[:3, 3]
                self.grasp_off = tool[2] - p[2]   # tool z above object center
                dest = self.task['dest']
                place = np.array([dest[0], dest[1], self.task['surf'] + he[2] + self.grasp_off + PLACE_EPS])
                rad = max(0.09, np.hypot(he[0], he[1]) + 0.015)
                pts = self._route(tool, place, n, max(self.grasp_off + he[2], 0.036), rad)
                self.path = self._mk(pts, self.task['yaw_place'])
                return None
            self.wait += 1
            if self.wait == 1:
                return self._act(grip=-1.0)
            self.wait = 0
            self.fail[('grasp', n)] = self.fail.get(('grasp', n), 0) + 1
            self.phase = 'plan'
            return None
        if ph == 'to_place':
            if not self.grasped:
                self.phase = 'plan'
                return None
            if blocked:
                self.phase = 'release'
                return None
            dq = self._track()
            if dq is None:
                self.phase = 'release'
                return None
            if self.last_final:
                self.phase = 'release'
                self.path = []
                self.wait = 1
                return self._act(dq, grip=1.0)
            return self._act(dq)
        if ph == 'release':
            if not self.grasped:
                self.phase = 'plan'
                self.wait = 0
                return None
            if False:
                tool = fk(self.base, self.q)[:3, 3]
                self.path = [(p_, self.task['yaw_place'] if self.task else 0.0)
                             for p_ in self._line(tool, np.array([tool[0], tool[1], tool[2] + 0.06]), 0.05)]
                self.wait = 0
                return None
            self.wait += 1
            if self.wait <= 2:
                return self._act(grip=1.0)
            # could not release: move down a little
            self.wait = 0
            tool = fk(self.base, self.q)[:3, 3]
            yaw = self.task['yaw_place'] if self.task else 0.0
            self.path = [(tool - np.array([0, 0, 0.003]), yaw)]
            dq = self._track()
            return self._act(dq, grip=1.0)
        if ph == 'retreat':
            dq = self._track()
            if dq is None or blocked:
                self.phase = 'plan'
                return None
            return self._act(dq)
        return self._act()
