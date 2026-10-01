"""Approach for kinder/PushPullHook2D-v0.

The robot base is confined to a low strip (y <= 1.15) while the two buttons
always sit above it, so the movable button can only be moved with the L-shaped
hook.  Plan:

  1. pick a push direction (close to M->T, exploiting the angular slack given
     by the 0.1 contact radius), the hook bar that will do the pushing, the
     hook orientation (bar perpendicular to the push direction, long bar
     hanging downwards so the grasp point is reachable) and the grasp point;
  2. drive the base to the grasp stand-off, extend the arm, touch, vacuum on;
  3. transit: translate at the current orientation to a spot where the whole
     rotation is collision free, rotate, then translate to the pre-push pose;
  4. push along the chosen direction until the buttons touch.

All planning uses an explicit geometric model of the environment that was
identified empirically (walls at x=0, x=3.5, y=0; base centre limited to
[0.1, 3.4] x [0.1, 1.15]; hook bar width extends inwards from the corner;
gripper front face at arm_joint + 0.005).
"""

import time
from collections import deque

import numpy as np

# ---------------------------------------------------------------- constants
DX_MAX = 0.05
DTH_MAX = 0.19634954
DARM_MAX = 0.1

ARM_MIN, ARM_MAX = 0.1, 0.2
BODY_R = 0.115             # true conservative footprint radius (base + gripper)
TRAVEL_R = 0.14            # radius used for planning masks (safety margin)

X_LO, X_HI = 0.102, 3.398  # base centre limits
Y_LO, Y_HI = 0.102, 1.148

WX_LO, WX_HI = 0.0, 3.5    # world walls for objects
WY_LO = 0.0

BR = 0.05                  # button radius
GRIP_FRONT = ARM_MAX + 0.005   # base centre -> gripper front face
PRE_OFF = 0.18             # pre-push back-off along -push_dir
RES = 0.035                # planning grid resolution
TIME_BUDGET = 32.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def perp(v):
    return np.array([-v[1], v[0]])


def dist_pt_rect(p, c, u, hl, hw):
    dx = p[0] - c[0]
    dy = p[1] - c[1]
    a = dx * u[0] + dy * u[1]
    b = -dx * u[1] + dy * u[0]
    ex = abs(a) - hl
    ey = abs(b) - hw
    if ex < 0.0:
        ex = 0.0
    if ey < 0.0:
        ey = 0.0
    return (ex * ex + ey * ey) ** 0.5


def rect_corners(c, u, hl, hw):
    w = np.array([-u[1], u[0]])
    return np.array([c + u * hl + w * hw, c + u * hl - w * hw,
                     c - u * hl + w * hw, c - u * hl - w * hw])


def offset_b(sig, w):
    """Signed offset along b of the base centre from the hook's C-line when the
    long bar is grasped from side ``sig`` (bar occupies b in [0, w])."""
    return (w + GRIP_FRONT) if sig > 0 else -GRIP_FRONT


class Hook(object):
    """corner C, orientation t.  long bar along a = -(cos t, sin t) with width
    extending along b = rot90(a); short bar along b with width along a."""

    def __init__(self, C, t, L1=1.25, L2=0.625, w=0.05):
        self.C = np.asarray(C, dtype=float)
        self.t = float(t)
        self.L1 = float(L1)
        self.L2 = float(L2)
        self.w = float(w)
        self.hw = float(w) * 0.5
        self.a = np.array([-np.cos(t), -np.sin(t)])
        self.b = perp(self.a)

    def bars(self):
        return [(self.C + self.a * (self.L1 * 0.5) + self.b * self.hw,
                 self.a, self.L1 * 0.5, self.hw),
                (self.C + self.b * (self.L2 * 0.5) + self.a * self.hw,
                 self.b, self.L2 * 0.5, self.hw)]

    def clear_of_point(self, p):
        return min(dist_pt_rect(p, *bar) for bar in self.bars())

    def in_world(self, margin=0.004):
        for c, u, hl, hw in self.bars():
            pts = rect_corners(c, u, hl, hw)
            if pts[:, 0].min() < WX_LO + margin or pts[:, 0].max() > WX_HI - margin:
                return False
            if pts[:, 1].min() < WY_LO + margin:
                return False
        return True


class GeneratedApproach(object):

    def __init__(self, action_space, observation_space, primitives=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.xs = np.arange(X_LO, X_HI + 1e-9, RES)
        self.ys = np.arange(Y_LO, Y_HI + 1e-9, RES)
        self.XX, self.YY = np.meshgrid(self.xs, self.ys, indexing='ij')

    # ------------------------------------------------------------------ api
    def reset(self, state, info=None):
        self.t0 = time.time()
        self.t = 0
        self.prev_state = None
        self.last_act = None
        self.stuck = 0
        self.blocked = set()
        self.queue = []
        self.phase = 'idle'
        self.grasp = None
        self.plan = None
        self.replans = 0
        self.parked = 0
        self.bad_grasp = 0
        self.bad_S = []
        self.hook_ref = None
        self.transit_replans = 0
        self._stall = 0
        self._last_prog = (-1, -9.9)
        self.touch_steps = 0
        self.phase_steps = 0
        self.goal_pose = (0.0, 0.0, 0.0)
        s = np.asarray(state, dtype=float)
        try:
            self._build_plan(s)
        except Exception:
            self.phase = 'idle'
        return None

    def get_action(self, state):
        s = np.asarray(state, dtype=float)
        self.t += 1
        try:
            if time.time() - self.t0 > TIME_BUDGET:
                self.phase = 'idle'
            a = self._act(s)
        except Exception:
            a = np.array([0.0, 0.0, 0.0, 0.0, 1.0 if s[6] > 0.5 else 0.0])
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.shape[0] != 5 or not np.all(np.isfinite(a)):
            a = np.zeros(5)
            a[4] = 1.0 if s[6] > 0.5 else 0.0
        lo = np.array([-DX_MAX, -DX_MAX, -DTH_MAX, -DARM_MAX, 0.0])
        hi = np.array([DX_MAX, DX_MAX, DTH_MAX, DARM_MAX, 1.0])
        a = np.clip(a, lo, hi)
        self.last_act = a
        self.prev_state = s.copy()
        return np.asarray(a, dtype=np.float32)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _hook_of(s):
        return Hook(s[9:11], s[11], s[18], s[19], s[17])

    def _robot_pose_for(self, C, t_hook):
        rel, dth = self.grasp
        thr = wrap(t_hook - dth)
        pos = np.asarray(C, dtype=float) - rot(thr).dot(rel)
        return pos, thr

    def _hook_for_robot(self, s, pos, thr):
        rel, dth = self.grasp
        C = np.asarray(pos, dtype=float) + rot(thr).dot(rel)
        return Hook(C, wrap(thr + dth), s[18], s[19], s[17])

    # ------------------------------------------------------------ planning
    def _candidates(self, s, held):
        M = s[20:22].copy()
        T = s[29:31].copy()
        L1, L2, W = float(s[18]), float(s[19]), float(s[17])
        dvec = T - M
        dist = float(np.linalg.norm(dvec))
        if dist < 1e-6:
            return [], np.array([1.0, 0.0])
        beta0 = np.arctan2(dvec[1], dvec[0])
        gmax = np.arcsin(min(1.0, 0.055 / dist))
        gammas = [0.0]
        for f in (0.35, 0.6, 0.8, 0.95):
            gammas.extend([f * gmax, -f * gmax])
        # wider fall-back angles: the push then only gets the button close and a
        # second push finishes the job (heavily penalised in the cost).
        for g in (0.25, 0.45, 0.65, 0.9, 1.15):
            if g > gmax + 0.02:
                gammas.extend([g, -g])
        hook0 = self._hook_of(s)
        a0, b0 = hook0.a, hook0.b
        out = []
        for gamma in gammas:
            beta = beta0 + gamma
            v = np.array([np.cos(beta), np.sin(beta)])
            need = dist * np.cos(gamma)
            resid = dist * abs(np.sin(gamma))
            extra = 0.0 if resid < 0.090 else (150.0 + 60.0 * resid)
            # a partial push that lifts the button higher makes the next round
            # harder (the base cannot go above y = 1.15)
            extra += 60.0 * max(0.0, need * v[1])
            for mode, alpha in (('long', beta + np.pi / 2), ('long', beta - np.pi / 2),
                                ('short', beta), ('short', beta + np.pi)):
                a = np.array([np.cos(alpha), np.sin(alpha)])
                if a[1] > -0.12:
                    continue
                b = perp(a)
                t_hook = wrap(alpha + np.pi)
                if mode == 'long':
                    sig_opts = [-1.0 if np.dot(v, b) > 0 else 1.0]
                    s_off = -BR if sig_opts[0] > 0 else (W + BR)
                else:
                    sig_opts = [1.0, -1.0]
                    s_off = -BR if np.dot(v, a) < 0 else (W + BR)
                prange = (np.arange(0.06, min(1.05, L1 - 0.12), 0.05) if mode == 'long'
                          else np.arange(0.05, L2 - 0.04, 0.05))
                drange = np.arange(0.15, L1 - 0.03, 0.05)
                for sig in sig_opts:
                    ob = offset_b(sig, W)
                    n0 = sig * b0
                    face0 = float(np.arctan2(-n0[1], -n0[0]))
                    for p in prange:
                        if mode == 'long':
                            C = M - p * a - s_off * b
                        else:
                            C = M - p * b - s_off * a
                        if held:
                            rp, _ = self._robot_pose_for(C, t_hook)
                            cfg = self._score(s, C, t_hook, rp, v, mode, None,
                                              sig, need, gamma)
                            if cfg is not None:
                                cfg['cost'] += extra
                                out.append(cfg)
                            continue
                        for d in drange:
                            if mode == 'long' and abs(d - p) < 0.13:
                                continue
                            rp = C + d * a + ob * b
                            S = hook0.C + a0 * d + b0 * (ob + sig * 0.055)
                            if not (X_LO + 0.015 <= S[0] <= X_HI - 0.015):
                                continue
                            if not (Y_LO + 0.02 <= S[1] <= Y_HI - 0.004):
                                continue
                            if dist_pt_rect(S, *hook0.bars()[1]) < TRAVEL_R + 0.02:
                                continue
                            if np.linalg.norm(S - M) < BR + TRAVEL_R + 0.05:
                                continue
                            cfg = self._score(s, C, t_hook, rp, v, mode, d,
                                              sig, need, gamma)
                            if cfg is not None:
                                cfg['cost'] += extra
                                cfg['S'] = S
                                cfg['face0'] = face0
                                cfg['n0'] = n0
                                cfg['cost'] += (np.linalg.norm(S - s[:2]) / DX_MAX
                                                + 6.0 * max(0.0, d - 0.9) / 0.05)
                                out.append(cfg)
        out.sort(key=lambda c: c['cost'])
        v0 = np.array([np.cos(beta0), np.sin(beta0)])
        return out, v0

    def _score(self, s, C, t_hook, rp, v, mode, d, sig, need, gamma):
        M = s[20:22]
        hook = Hook(C, t_hook, s[18], s[19], s[17])
        if not hook.in_world():
            return None
        bars = hook.bars()
        idx = 0 if mode == 'long' else 1
        if dist_pt_rect(M, *bars[1 - idx]) < BR + 0.012:
            return None
        if not (X_LO + 0.004 <= rp[0] <= X_HI - 0.004
                and Y_LO + 0.004 <= rp[1] <= Y_HI - 0.004):
            return None
        if np.linalg.norm(rp - M) < BR + TRAVEL_R:
            return None
        C2 = C + v * (need + 0.02)
        rp2 = rp + v * (need + 0.02)
        h2 = Hook(C2, t_hook, s[18], s[19], s[17])
        if not h2.in_world():
            return None
        if not (X_LO <= rp2[0] <= X_HI and Y_LO <= rp2[1] <= Y_HI):
            return None
        if np.linalg.norm(rp2 - M) < BR + TRAVEL_R:
            return None
        pre_off = None
        for cand_off in (PRE_OFF, 0.13, 0.09, 0.06):
            good = True
            for off in (-cand_off, -0.55 * cand_off):
                C2 = C + v * off
                rp2 = rp + v * off
                h2 = Hook(C2, t_hook, s[18], s[19], s[17])
                if not h2.in_world():
                    good = False
                    break
                if not (X_LO <= rp2[0] <= X_HI and Y_LO <= rp2[1] <= Y_HI):
                    good = False
                    break
                if np.linalg.norm(rp2 - M) < BR + TRAVEL_R:
                    good = False
                    break
                if dist_pt_rect(M, *h2.bars()[1 - idx]) < BR + 0.012:
                    good = False
                    break
                if dist_pt_rect(M, *h2.bars()[idx]) < BR + 0.012:
                    good = False
                    break
            if good:
                pre_off = cand_off
                break
        if pre_off is None:
            return None
        cost = (abs(wrap(t_hook - s[11])) / DTH_MAX
                + 0.4 * abs(rp[1] - 0.95) / DX_MAX
                + 40.0 * abs(gamma))
        return dict(mode=mode, C=np.asarray(C, dtype=float), t_hook=float(t_hook),
                    rp=np.asarray(rp, dtype=float), v=v, d=d, sig=sig,
                    need=float(need), gamma=float(gamma), pre_off=float(pre_off),
                    resid=float(np.linalg.norm(s[29:31] - M) * abs(np.sin(gamma))),
                    park=False,
                    cost=float(cost) + 3.0 * (PRE_OFF - pre_off) / 0.05)

    def _build_plan(self, s):
        cands, v0 = self._candidates(s, held=False)
        hook = self._hook_of(s)
        mask = self._base_mask(s, hook)
        comp = self._component(mask, self._cell(s[:2]))
        chosen = None
        for cfg in cands:
            i, j = self._cell(cfg['S'])
            if not comp[i, j]:
                continue
            if any(np.linalg.norm(cfg['S'] - b) < 0.06 for b in self.bad_S):
                continue
            chosen = cfg
            break
        if (chosen is None or chosen['resid'] >= 0.090) and self.parked < 2:
            # the hook cuts the workspace in two: park it out of the way first
            park = self._park_plan(s, comp)
            if park is not None:
                chosen = park
                self.parked += 1
        if chosen is None:
            if not cands:
                self.phase = 'idle'
                self.plan = None
                return
            chosen = cands[0]
        self.plan = chosen
        pts = self._path(mask, self._cell(s[:2]), self._cell(chosen['S']))
        self.queue = pts + [tuple(chosen['S'])]
        self.grasp_target = (chosen['S'], chosen['face0'], chosen['n0'])
        self.phase = 'goto_grasp'
        self.phase_steps = 0

    def _park_plan(self, s, comp):
        """Move the hook to a vertical pose whose lower tip is high enough that
        the base can drive underneath, so both sides become reachable."""
        hook0 = self._hook_of(s)
        a0, b0 = hook0.a, hook0.b
        L1, W = float(s[18]), float(s[17])
        M = s[20:22]
        t_park = np.pi / 2
        ap = np.array([0.0, -1.0])
        bp = np.array([1.0, 0.0])
        best = None
        for sig in (1.0, -1.0):
            ob = offset_b(sig, W)
            n0 = sig * b0
            face0 = float(np.arctan2(-n0[1], -n0[0]))
            for d in np.arange(0.55, min(1.2, L1 - 0.05), 0.05):
                S = hook0.C + a0 * d + b0 * (ob + sig * 0.055)
                if not (X_LO + 0.015 <= S[0] <= X_HI - 0.015):
                    continue
                if not (Y_LO + 0.02 <= S[1] <= Y_HI - 0.004):
                    continue
                if dist_pt_rect(S, *hook0.bars()[1]) < TRAVEL_R + 0.02:
                    continue
                if np.linalg.norm(S - M) < BR + TRAVEL_R + 0.05:
                    continue
                ci, cj = self._cell(S)
                if not comp[ci, cj]:
                    continue
                for cy in (1.62, 1.70, 1.55):
                    for xp in np.arange(0.35, 3.2, 0.1):
                        C = np.array([xp, cy])
                        rp = C + d * ap + ob * bp
                        if not (X_LO + 0.01 <= rp[0] <= X_HI - 0.01
                                and Y_LO + 0.01 <= rp[1] <= Y_HI - 0.01):
                            continue
                        hk = Hook(C, t_park, s[18], s[19], s[17])
                        if not hk.in_world(0.02):
                            continue
                        if hk.clear_of_point(M) < BR + 0.08:
                            continue
                        if np.linalg.norm(rp - M) < BR + TRAVEL_R + 0.03:
                            continue
                        cost = (np.linalg.norm(S - s[:2]) / DX_MAX
                                + abs(wrap(t_park - s[11])) / DTH_MAX
                                - 3.0 * min(abs(xp - M[0]), 1.6))
                        if best is None or cost < best[0]:
                            best = (cost, dict(mode='long', C=C, t_hook=t_park,
                                               rp=rp, v=np.zeros(2), d=d, sig=sig,
                                               need=0.0, gamma=0.0, pre_off=0.0,
                                               resid=9.0, park=True, cost=cost,
                                               S=S, face0=face0, n0=n0))
        return None if best is None else best[1]

    def _build_plan_held(self, s):
        cands, v0 = self._candidates(s, held=True)
        seen = set()
        tried = 0
        for cfg in cands:
            key = (cfg['mode'], round(cfg['t_hook'], 2), cfg['sig'],
                   round(float(cfg['rp'][0]), 1), round(float(cfg['rp'][1]), 1))
            if key in seen:
                continue
            seen.add(key)
            tried += 1
            if tried > 22 or time.time() - self.t0 > TIME_BUDGET:
                break
            self.plan = cfg
            if self._start_transit(s):
                return True
        # last resort: aim straight at the goal pose and let the reactive
        # servo deal with obstacles
        for cfg in cands[:1]:
            self.plan = cfg
            gpos, gthr = self._robot_pose_for(cfg['C'], cfg['t_hook'])
            pre = gpos - cfg['v'] * cfg.get('pre_off', PRE_OFF)
            pre[0] = float(np.clip(pre[0], X_LO, X_HI))
            pre[1] = float(np.clip(pre[1], Y_LO, Y_HI))
            self.final_pos = gpos
            self.goal_thr = gthr
            self.pre_pose = pre
            self.legs = [('t', [(float(pre[0]), float(pre[1]))], gthr)]
            self.leg_i = 0
            self.phase = 'transit'
            self.phase_steps = 0
            return True
        return False

    # ---------------------------------------------------------------- grids
    def _cell(self, p):
        i = int(round((p[0] - self.xs[0]) / RES))
        j = int(round((p[1] - self.ys[0]) / RES))
        return (min(max(i, 0), len(self.xs) - 1), min(max(j, 0), len(self.ys) - 1))

    def _base_mask(self, s, hook):
        """Free cells for the base while the hook lies untouched."""
        free = np.ones(self.XX.shape, dtype=bool)
        for c, u, hl, hw in hook.bars():
            dx = self.XX - c[0]
            dy = self.YY - c[1]
            aa = dx * u[0] + dy * u[1]
            bb = -dx * u[1] + dy * u[0]
            ex = np.maximum(np.abs(aa) - hl, 0.0)
            ey = np.maximum(np.abs(bb) - hw, 0.0)
            free &= (np.hypot(ex, ey) > TRAVEL_R)
        free &= (np.hypot(self.XX - s[20], self.YY - s[21]) > BR + TRAVEL_R)
        return free

    def _carry_mask(self, s, thr, touch_ok=False, pad=0.030):
        """Free base cells while carrying the hook at robot angle thr."""
        rel, dth = self.grasp
        off = rot(thr).dot(rel)
        hk = Hook(off, wrap(thr + dth), s[18], s[19], s[17])  # corner relative
        M = s[20:22]
        free = np.ones(self.XX.shape, dtype=bool)
        for c, u, hl, hw in hk.bars():
            if not touch_ok:
                dx = M[0] - (c[0] + self.XX)
                dy = M[1] - (c[1] + self.YY)
                aa = dx * u[0] + dy * u[1]
                bb = -dx * u[1] + dy * u[0]
                ex = np.maximum(np.abs(aa) - hl, 0.0)
                ey = np.maximum(np.abs(bb) - hw, 0.0)
                free &= (np.hypot(ex, ey) > BR + pad)
            cor = rect_corners(c, u, hl, hw)
            wall = 0.020 if pad > 0.02 else 0.008
            free &= (self.XX + cor[:, 0].min() > WX_LO + wall)
            free &= (self.XX + cor[:, 0].max() < WX_HI - wall)
            free &= (self.YY + cor[:, 1].min() > WY_LO + wall)
        free &= (np.hypot(self.XX - M[0], self.YY - M[1]) > BR + TRAVEL_R)
        return free

    @staticmethod
    def _component(mask, start):
        if not mask[start[0], start[1]]:
            alt = GeneratedApproach._nearest_cell(mask, start)
            if alt is None:
                return np.zeros_like(mask)
            start = alt
        d = GeneratedApproach._bfs_dist(mask, start)
        return d >= 0

    @staticmethod
    def _bfs_dist(mask, start):
        nx, ny = mask.shape
        dist = np.full(mask.shape, -1, dtype=np.int32)
        si, sj = start
        dq = deque()
        dist[si, sj] = 0
        dq.append((si, sj))
        nb = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
        while dq:
            i, j = dq.popleft()
            di = dist[i, j] + 1
            for u, v in nb:
                a = i + u
                b = j + v
                if 0 <= a < nx and 0 <= b < ny and mask[a, b] and dist[a, b] < 0:
                    dist[a, b] = di
                    dq.append((a, b))
        return dist

    def _path(self, mask, start, goal):
        m = mask.copy()
        if not m[start[0], start[1]]:
            alt = self._nearest_cell(m, start)
            if alt is not None and abs(alt[0] - start[0]) + abs(alt[1] - start[1]) <= 6:
                m[start[0], start[1]] = True
                si, sj = start
                m[min(si, alt[0]):max(si, alt[0]) + 1,
                  min(sj, alt[1]):max(sj, alt[1]) + 1] = True
        m[start[0], start[1]] = True
        m[goal[0], goal[1]] = True
        dist = self._bfs_dist(m, goal)
        if dist[start[0], start[1]] < 0:
            return []
        pts = []
        i, j = start
        nb = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
        guard = 0
        while (i, j) != goal and guard < 4000:
            guard += 1
            best = None
            bd = dist[i, j]
            for u, v in nb:
                a, b = i + u, j + v
                if 0 <= a < m.shape[0] and 0 <= b < m.shape[1] \
                        and dist[a, b] >= 0 and dist[a, b] < bd:
                    bd = dist[a, b]
                    best = (a, b)
            if best is None:
                break
            i, j = best
            pts.append((float(self.xs[i]), float(self.ys[j])))
        return [pts[k] for k in range(2, len(pts), 3)]

    # ------------------------------------------------------------- transit
    def _start_transit(self, s):
        """Plan a translate/rotate sequence to the pre-push pose.

        The hook orientation is swept monotonically from the current angle to
        the goal angle.  For every intermediate angle the set of collision-free
        base cells is computed and split into connected components; a path is
        searched through the (angle, component) graph, which allows the base to
        drive around while the hook turns.
        """
        pl = self.plan
        gpos, gthr = self._robot_pose_for(pl['C'], pl['t_hook'])
        pre = gpos - pl['v'] * pl.get('pre_off', PRE_OFF)
        pre[0] = float(np.clip(pre[0], X_LO, X_HI))
        pre[1] = float(np.clip(pre[1], Y_LO, Y_HI))
        self.final_pos = gpos
        self.goal_thr = gthr
        self.pre_pose = pre
        thr0 = float(s[2])
        start = self._cell(s[:2])
        goal = self._cell(pre)
        best = None
        for pad in (0.030, 0.010):
            for direction in (0, 1):
                delta = float(wrap(gthr - thr0))
                if direction == 1:
                    if abs(delta) < 1e-6:
                        continue
                    delta = delta - 2 * np.pi * np.sign(delta)
                legs = self._layered_plan(s, thr0, delta, start, goal, pad)
                if legs is None:
                    continue
                cost = (sum(len(l[1]) for l in legs if l[0] == 't') * 1.4
                        + abs(delta) / DTH_MAX)
                if best is None or cost < best[0]:
                    best = (cost, legs)
            if best is not None:
                break
        if False:
            legs = None
        if best is None:
            return False
        self.legs = list(best[1])
        self.leg_i = 0
        self.phase = 'transit'
        self.phase_steps = 0
        return True

    def _layered_plan(self, s, thr0, delta, start, goal, pad=0.030):
        try:
            from scipy import ndimage
        except Exception:
            ndimage = None
        K = max(1, int(np.ceil(abs(delta) / 0.06)))
        thetas = [thr0 + delta * k / K for k in range(K + 1)]
        masks = []
        for th in thetas:
            masks.append(self._carry_mask(s, th, pad=pad))
        masks[0][start[0], start[1]] = True
        gi, gj = goal
        masks[K][max(gi - 1, 0):gi + 2, max(gj - 1, 0):gj + 2] = True
        si, sj = start
        masks[0][max(si - 1, 0):si + 2, max(sj - 1, 0):sj + 2] = True
        if ndimage is None:
            return None
        st = np.ones((3, 3), dtype=bool)
        labs = []
        for m in masks:
            lab, _ = ndimage.label(m, structure=st)
            labs.append(lab)
        c0 = int(labs[0][start[0], start[1]])
        cg = int(labs[K][goal[0], goal[1]])
        if c0 == 0 or cg == 0:
            return None
        # component graph over layers
        edges = []
        for k in range(K):
            both = (labs[k] > 0) & (labs[k + 1] > 0)
            if not both.any():
                edges.append({})
                continue
            a = labs[k][both].astype(np.int64)
            b = labs[k + 1][both].astype(np.int64)
            pairs = np.unique(a * 100000 + b)
            e = {}
            for pv in pairs:
                ca = int(pv // 100000)
                cb = int(pv % 100000)
                e.setdefault(ca, []).append(cb)
            edges.append(e)
        # BFS over (layer, comp)
        par = {(0, c0): None}
        dq = deque([(0, c0)])
        target = None
        while dq:
            k, c = dq.popleft()
            if k == K and c == cg:
                target = (k, c)
                break
            if k >= K:
                continue
            for cb in edges[k].get(c, ()):
                if (k + 1, cb) not in par:
                    par[(k + 1, cb)] = (k, c)
                    dq.append((k + 1, cb))
        if target is None:
            return None
        chain = []
        node = target
        while node is not None:
            chain.append(node)
            node = par[node]
        chain.reverse()
        # reconstruct waypoints
        legs = []
        cur = start
        for k in range(K):
            ck = chain[k][1]
            ck1 = chain[k + 1][1]
            inter = (labs[k] == ck) & (labs[k + 1] == ck1)
            if not inter[cur[0], cur[1]]:
                cell = self._nearest_cell(inter, cur)
                if cell is None:
                    return None
                pts = self._path(masks[k], cur, cell)
                legs.append(('t', pts + [(float(self.xs[cell[0]]),
                                          float(self.ys[cell[1]]))], thetas[k]))
                cur = cell
            legs.append(('r', thetas[k + 1], (float(self.xs[cur[0]]),
                                              float(self.ys[cur[1]]))))
        pts = self._path(masks[K], cur, goal)
        legs.append(('t', pts + [(float(self.pre_pose[0]), float(self.pre_pose[1]))],
                     thetas[K]))
        # merge consecutive rotations
        merged = []
        for leg in legs:
            if leg[0] == 'r' and merged and merged[-1][0] == 'r':
                merged[-1] = ('r', leg[1], merged[-1][2])
            elif leg[0] == 't' and not leg[1]:
                continue
            else:
                merged.append(leg)
        return merged

    @staticmethod
    def _nearest_cell(mask, cur):
        idx = np.argwhere(mask)
        if len(idx) == 0:
            return None
        d = (idx[:, 0] - cur[0]) ** 2 + (idx[:, 1] - cur[1]) ** 2
        k = int(np.argmin(d))
        return (int(idx[k, 0]), int(idx[k, 1]))

    # ------------------------------------------------------------ execution
    def _act(self, s):
        if self.prev_state is not None:
            if np.abs(s - self.prev_state).max() < 1e-7:
                self.stuck += 1
                if self.last_act is not None:
                    self.blocked.add(self._key(self.last_act))
            else:
                self.stuck = 0
                self.blocked = set()
        if self.grasp is not None and self.phase in ('transit', 'push'):
            hk = self._hook_for_robot(s, s[:2], s[2])
            if (np.abs(hk.C - s[9:11]).max() > 0.02
                    or abs(wrap(hk.t - s[11])) > 0.05):
                # the vacuum never actually caught the hook (or lost it)
                self.grasp = None
                self.blocked = set()
                self.stuck = 0
                self.bad_grasp += 1
                if self.bad_grasp <= 4:
                    self._build_plan(s)
                else:
                    self.phase = 'idle'
                if self.phase == 'idle':
                    return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.phase_steps += 1
        ph = self.phase
        if ph == 'goto_grasp':
            return self._ph_goto_grasp(s)
        if ph == 'touch':
            return self._ph_touch(s)
        if ph == 'transit':
            return self._ph_transit(s)
        if ph == 'release':
            return self._ph_release(s)
        if ph == 'push':
            return self._ph_push(s)
        return np.array([0.0, 0.0, 0.0, 0.0, 1.0 if s[6] > 0.5 else 0.0])

    @staticmethod
    def _key(a):
        return (round(float(a[0]), 4), round(float(a[1]), 4), round(float(a[2]), 4))

    def _direct(self, s, gx, gy, gth, arm, vac):
        return np.array([np.clip(gx - s[0], -DX_MAX, DX_MAX),
                         np.clip(gy - s[1], -DX_MAX, DX_MAX),
                         np.clip(wrap(gth - s[2]), -DTH_MAX, DTH_MAX),
                         np.clip(arm - s[4], -DARM_MAX, DARM_MAX),
                         vac])

    def _valid(self, s, dx, dy, dth, mode):
        nx = s[0] + dx
        ny = s[1] + dy
        if not (X_LO <= nx <= X_HI and Y_LO <= ny <= Y_HI):
            return False
        M = s[20:22]
        p = np.array([nx, ny])
        cm = np.hypot(s[0] - M[0], s[1] - M[1])
        if np.hypot(nx - M[0], ny - M[1]) < min(BR + BODY_R, cm - 1e-9):
            return False
        if mode == 'free':
            hook = self._hook_of(s)
            bars = hook.bars()
            cur = min(dist_pt_rect(s[:2], *bar) for bar in bars)
            new = min(dist_pt_rect(p, *bar) for bar in bars)
            thresh = BODY_R if cur >= BODY_R else cur - 1e-9
            if new < thresh:
                return False
            return True
        fracs = (0.34, 0.67, 1.0) if abs(dth) > 0.05 else (1.0,)
        for f in fracs:
            hk = self._hook_for_robot(s, s[:2] + np.array([dx * f, dy * f]),
                                      s[2] + dth * f)
            if not hk.in_world():
                return False
            if mode != 'push' and hk.clear_of_point(M) < BR + 0.006:
                return False
        return True

    def _servo(self, s, gx, gy, gth, arm, vac, mode, wang=1.0):
        act = self._direct(s, gx, gy, gth, arm, vac)
        if self._key(act) not in self.blocked and self._valid(s, act[0], act[1], act[2], mode):
            return act
        best = None
        bs = 1e18
        da = float(np.clip(arm - s[4], -DARM_MAX, DARM_MAX))
        opts = []
        for fx in (1.0, 0.5, 0.0, -0.5, -1.0):
            for fy in (1.0, 0.5, 0.0, -0.5, -1.0):
                for ft in (1.0, 0.5, 0.0, -1.0):
                    opts.append((act[0] * fx, act[1] * fy, act[2] * ft))
        for ux in (-DX_MAX, -DX_MAX * 0.5, 0.0, DX_MAX * 0.5, DX_MAX):
            for uy in (-DX_MAX, -DX_MAX * 0.5, 0.0, DX_MAX * 0.5, DX_MAX):
                opts.append((ux, uy, 0.0))
        for c in opts:
            if abs(c[0]) < 1e-9 and abs(c[1]) < 1e-9 and abs(c[2]) < 1e-9:
                continue
            if self._key((c[0], c[1], c[2])) in self.blocked:
                continue
            if not self._valid(s, c[0], c[1], c[2], mode):
                continue
            sc = (np.hypot(gx - s[0] - c[0], gy - s[1] - c[1])
                  + wang * abs(wrap(gth - s[2] - c[2])))
            if sc < bs:
                bs = sc
                best = c
        if best is None:
            return np.array([0.0, 0.0, 0.0, da, vac])
        return np.array([best[0], best[1], best[2], da, vac])

    # ------------------------------------------------------------- phases
    def _ph_goto_grasp(self, s):
        S, face, n0 = self.grasp_target
        while self.queue and np.hypot(self.queue[0][0] - s[0],
                                      self.queue[0][1] - s[1]) < 0.03:
            self.queue.pop(0)
        near = np.hypot(S[0] - s[0], S[1] - s[1])
        if not self.queue:
            if near < 4e-3 and abs(wrap(face - s[2])) < 3e-3 and abs(s[4] - ARM_MAX) < 1e-3:
                self.phase = 'touch'
                self.touch_steps = 0
                self.hook_ref = None
                return self._ph_touch(s)
            arm = ARM_MAX if near < 0.10 else ARM_MIN
            return self._servo(s, S[0], S[1], face, arm, 0.0, 'free')
        if self.phase_steps > 260:
            self.phase_steps = 0
            self.blocked = set()
            self.replans += 1
            if self.replans > 8:
                self.phase = 'idle'
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.grasp = None
            self._build_plan(s)
            return self._act(s)
        wp = self.queue[0]
        return self._servo(s, wp[0], wp[1], face, ARM_MIN, 0.0, 'free')

    def _ph_touch(self, s):
        """Creep into the bar with the vacuum already on.

        The hook can only move if it is attached (the base alone cannot push
        it), so a change of the hook pose is a reliable grasp detector.
        """
        S, face, n0 = self.grasp_target
        if abs(s[4] - ARM_MAX) > 1e-3:
            return np.array([0.0, 0.0, 0.0, ARM_MAX - s[4], 1.0])
        if self.hook_ref is None:
            self.hook_ref = s[9:12].copy()
        if np.abs(s[9:12] - self.hook_ref).max() > 1e-6:
            thr = float(s[2])
            self.grasp = (rot(-thr).dot(s[9:11] - s[:2]), wrap(s[11] - thr))
            if not self._start_transit(s):
                if not self._build_plan_held(s):
                    self.grasp = None
                    self._fail_grasp(s)
            return self._act(s)
        self.touch_steps += 1
        if self.touch_steps > 14:
            self._fail_grasp(s)
            return self._act(s) if self.phase != 'idle' else \
                np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        return np.array([-n0[0] * 0.012, -n0[1] * 0.012, 0.0, 0.0, 1.0])

    def _fail_grasp(self, s):
        """The chosen grasp did not work; blacklist it and plan another one."""
        self.bad_grasp += 1
        self.blocked = set()
        self.stuck = 0
        self.grasp = None
        try:
            self.bad_S.append(np.asarray(self.grasp_target[0], dtype=float))
        except Exception:
            pass
        if self.bad_grasp > 4:
            self.phase = 'idle'
            return
        self._build_plan(s)

    def _ph_transit(self, s):
        # progress = how far through the leg list we are, then how close to the
        # current way point; stall if that has not improved for a while
        if self.leg_i < len(self.legs) and self.legs[self.leg_i][0] == 't' \
                and self.legs[self.leg_i][1]:
            wp = self.legs[self.leg_i][1][0]
            rem = np.hypot(wp[0] - s[0], wp[1] - s[1])
        elif self.leg_i < len(self.legs):
            rem = abs(wrap(self.legs[self.leg_i][1] - s[2]))
        else:
            rem = np.hypot(self.pre_pose[0] - s[0], self.pre_pose[1] - s[1])
        prog = (self.leg_i, -round(float(rem), 3))
        if prog <= getattr(self, '_last_prog', (-1, -9.9)):
            self._stall = getattr(self, '_stall', 0) + 1
        else:
            self._stall = 0
            self._last_prog = prog
        if self._stall > 30:
            self._stall = 0
            self.transit_replans = getattr(self, 'transit_replans', 0) + 1
            self.blocked = set()
            if self.transit_replans <= 4 and self._start_transit(s):
                return self._act(s)
            self._replan(s)
            return self._act(s) if self.phase != 'idle' else \
                np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        if self.phase_steps > 220:
            self._replan(s)
            return self._act(s) if self.phase != 'idle' else \
                np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        while self.leg_i < len(self.legs):
            kind, data = self.legs[self.leg_i][0], self.legs[self.leg_i][1]
            if kind == 'r':
                if abs(wrap(data - s[2])) < 3e-3:
                    self.leg_i += 1
                    continue
                hold = self.legs[self.leg_i][2]
                return self._servo(s, hold[0], hold[1], data, ARM_MAX, 1.0,
                                   'carry', wang=10.0)
            pts = data
            th = self.legs[self.leg_i][2]
            nxt_rot = (self.leg_i + 1 < len(self.legs)
                       and self.legs[self.leg_i + 1][0] == 'r')
            last = (self.leg_i == len(self.legs) - 1)
            tol = 0.006 if last else (0.008 if nxt_rot and len(pts) == 1 else 0.03)
            while pts and np.hypot(pts[0][0] - s[0], pts[0][1] - s[1]) < tol:
                pts.pop(0)
            if not pts:
                self.leg_i += 1
                continue
            return self._servo(s, pts[0][0], pts[0][1], th, ARM_MAX, 1.0, 'carry')
        gx, gy = self.pre_pose
        if np.hypot(gx - s[0], gy - s[1]) < 6e-3 and abs(wrap(self.goal_thr - s[2])) < 4e-3:
            if self.plan.get('park'):
                self.phase = 'release'
                self.phase_steps = 0
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.phase = 'push'
            self.phase_steps = 0
            self.push_travel = 0.0
            self.push_budget = self.plan.get('pre_off', PRE_OFF) + self.plan['need'] + 0.14
            return self._ph_push(s)
        return self._servo(s, gx, gy, self.goal_thr, ARM_MAX, 1.0, 'carry')

    def _ph_release(self, s):
        if s[6] > 0.5:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.grasp = None
        self.blocked = set()
        self.stuck = 0
        self._build_plan(s)
        if self.phase == 'idle':
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        return self._act(s)

    def _ph_push(self, s):
        """Advance along the push direction.

        The simulator resolves a hook/button overlap by *doubling* the
        remaining centre-to-rectangle distance, so the button only tracks the
        hook if each step is at most ~0.025; larger steps make the bar tunnel
        straight through the button.  Approach fast, push slowly.
        """
        pl = self.plan
        v = pl['v']
        d = float(np.linalg.norm(s[29:31] - s[20:22]))
        if d < 0.085:
            return np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        if self.push_travel > self.push_budget or self.stuck >= 4:
            self._replan(s)
            return self._act(s) if self.phase != 'idle' else \
                np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        hk = self._hook_for_robot(s, s[:2], s[2])
        idx = 0 if pl['mode'] == 'long' else 1
        # q = centre-to-rectangle distance.  Contact fires below BR = 0.05 and
        # the simulator then doubles q, so the fastest 1:1 tracking cycle is
        # q: 0.098 -> 0.049 -> 0.098, i.e. always step to q - 0.049.
        q = dist_pt_rect(s[20:22], *hk.bars()[idx])
        if q < 0.040:
            self._replan(s)
            return self._act(s) if self.phase != 'idle' else \
                np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        step = float(np.clip(q - 0.049, 0.004, DX_MAX))
        if not self._valid(s, v[0] * step, v[1] * step, 0.0, 'push'):
            step = 0.5 * step
            if not self._valid(s, v[0] * step, v[1] * step, 0.0, 'push'):
                self._replan(s)
                return self._act(s) if self.phase != 'idle' else \
                    np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        self.push_travel += step
        return np.array([v[0] * step, v[1] * step, 0.0, 0.0, 1.0])

    def _replan(self, s):
        self.replans += 1
        self.stuck = 0
        self.blocked = set()
        if self.replans > 6 or time.time() - self.t0 > TIME_BUDGET:
            self.phase = 'idle'
            return
        if self._build_plan_held(s):
            return
        # give up on the current grasp: release and start over
        self.grasp = None
        self._build_plan(s)
        if self.phase == 'goto_grasp':
            self.queue.insert(0, (float(s[0]), float(s[1])))
