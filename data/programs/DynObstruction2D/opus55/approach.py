import copy
import numpy as np

LIM = np.array([0.049, 0.049, 0.19, 0.099, 0.0199])
Y_TRAVEL = 1.40
Y_PUSH = 0.80
ARM_MIN = 0.24
ARM_MAX = 0.48
GAP_OPEN = 0.32
GAP_MIN = 0.12
PUSH_OFF = 0.195      # contact offset: pusher x to contact face (gap open, arm down)
CLEAR_HALF = 0.26     # half-width of region occupied by gripper/base when descending
WALL_L = 0.0
WALL_R = 3.236
X_MIN = 0.2405
X_MAX = WALL_R - 0.2405
FLOOR = 0.1
FTIP = 0.205          # fingertip below arm end
GRASP_W = 0.255      # max graspable width
DOWN = -np.pi / 2


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def tilt(th):
    """distance of angle to nearest multiple of pi/2"""
    m = th % (np.pi / 2)
    return min(m, np.pi / 2 - m)


class Obj:
    def __init__(self, state, o):
        self.name = o.name
        self.x = state.get(o, 'x')
        self.y = state.get(o, 'y')
        self.th = state.get(o, 'theta')
        self.w = state.get(o, 'width')
        self.h = state.get(o, 'height')
        try:
            self.held = state.get(o, 'held') > 0.5
        except Exception:
            self.held = False
        c, s = np.cos(self.th), np.sin(self.th)
        xs, ys = [], []
        for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            px = dx * self.w / 2
            py = dy * self.h / 2
            xs.append(self.x + c * px - s * py)
            ys.append(self.y + s * px + c * py)
        self.xl, self.xr = min(xs), max(xs)
        self.yb, self.yt = min(ys), max(ys)
        self.cx = 0.5 * (self.xl + self.xr)
        self.aw = self.xr - self.xl
        self.ah = self.yt - self.yb


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.reset(None, None)

    def reset(self, state, info):
        self.phase = 'init'
        self.task = None
        self.stuck = 0
        self.prev_y = None
        self.t = 0
        self.fails = 0
        self.grasp_off = None
        self._block_moved = 0
        self.wedges = 0
        self.px_extra = {}
        self.wedge_gap = {}
        self.no_wedge = False
        self.rot_target = None

    # ------------------------------------------------------------------
    def _parse(self, state):
        rob = None
        block = None
        surf = None
        obstacles = []
        for n in state.get_object_names():
            o = state.get_object_from_name(n)
            tname = o.type.name
            if tname == 'kin_robot':
                rob = o
            elif tname == 'target_block':
                block = Obj(state, o)
            elif tname == 'target_surface':
                surf = Obj(state, o)
            else:
                try:
                    obstacles.append(Obj(state, o))
                except Exception:
                    pass
        r = np.array([state.get(rob, f) for f in
                      ['x', 'y', 'theta', 'arm_joint', 'finger_gap']])
        return r, block, surf, obstacles

    def _goto(self, r, target, tol=2e-3):
        t = np.array([r[k] if target[k] is None else target[k] for k in range(5)])
        d = t - r
        d[2] = wrap(d[2])
        return np.clip(d, -LIM, LIM), bool(np.all(np.abs(d) < tol))

    # ------------------------------------------------------------------
    def _goal_range(self, block, surf):
        bw = block.aw
        sl, sr = surf.x - surf.w / 2, surf.x + surf.w / 2
        slack = (sr - sl) - bw
        m = min(0.015, max(slack / 4, 0.0))
        return sl + bw / 2 + m, sr - bw / 2 - m

    def _chain_end(self, g, d, bw, bx, obstacles):
        edge = g + d * bw / 2
        ahead = [o for o in obstacles if d * (o.cx - bx) > 0 and o.yb < FLOOR + 0.3]
        ahead.sort(key=lambda o: d * o.cx)
        for o in ahead:
            nearo = o.xl if d > 0 else o.xr
            if d * (nearo - edge) < 0:
                nearo = edge
            edge = nearo + d * o.aw
        return edge

    def _chain_ok(self, g, d, bw, bx, obstacles):
        e = self._chain_end(g, d, bw, bx, obstacles)
        return e < WALL_R - 0.02 if d > 0 else e > WALL_L + 0.02

    def _free_spot(self, w, obj_name, block, surf, obstacles, forbid, prefer_x, all_spots=False):
        """find floor x where an object of width w can be placed (with finger clearance)"""
        best = None
        cands = []
        half = w / 2 + 0.1
        for x in np.arange(0.3, WALL_R - 0.3 + 1e-9, 0.02):
            lo, hi = x - half, x + half
            if lo < WALL_L + 0.02 or hi > WALL_R - 0.02:
                continue
            ok = True
            for (fl, fr) in forbid:
                if hi > fl and lo < fr:
                    ok = False
                    break
            if not ok:
                continue
            for o in obstacles + [block]:
                if o.name == obj_name:
                    continue
                if o.xr + 0.02 > lo and o.xl - 0.02 < hi:
                    ok = False
                    break
            if not ok:
                continue
            c = abs(x - prefer_x) + 2.0 * max(0.0, 0.6 - min(lo - WALL_L, WALL_R - hi))
            cands.append((c, float(x)))
            if best is None or c < best[0]:
                best = (c, x)
        if all_spots:
            return [x for c, x in sorted(cands)]
        return None if best is None else best[1]

    def _sim_push(self, o, d, dist, objs):
        """1D simulate pushing o by dist in direction d; return (chain_end, moved names)"""
        edge = (o.xr if d > 0 else o.xl) + d * dist
        moved = [o.name]
        ahead = [p for p in objs if p.name != o.name and d * (p.cx - o.cx) > 0
                 and p.yb < FLOOR + 0.3]
        ahead.sort(key=lambda p: d * p.cx)
        for p in ahead:
            nearp = p.xl if d > 0 else p.xr
            if d * (nearp - edge) < 0:
                moved.append(p.name)
                edge = edge + d * p.aw
            else:
                edge = p.xr if d > 0 else p.xl
        return edge, moved

    def _clear_push(self, o, zl, zr, block, obstacles):
        """push obstacle o out of zone [zl,zr] without moving the block"""
        best = None
        allo = obstacles + [block]
        for d in (1.0, -1.0):
            tgt = (zr + o.aw / 2 + 0.02) if d > 0 else (zl - o.aw / 2 - 0.02)
            dist = d * (tgt - o.cx)
            if dist < 0:
                continue
            end, moved = self._sim_push(o, d, dist, allo)
            if block.name in moved:
                continue
            if (d > 0 and end > WALL_R - 0.02) or (d < 0 and end < WALL_L + 0.02):
                continue
            t = self._push_task(o, d, tgt, allo, protect=block.name, strict=True)
            if t is None:
                continue
            bad = False
            for p in allo:
                if p.name in t['chain'] and p.name != o.name:
                    if p.xr + d * dist > zl and p.xl + d * dist < zr:
                        bad = True
            if bad:
                continue
            if best is None or dist < best[0]:
                best = (dist, t)
        if best is None:
            return None
        return best[1]

    def _plan(self, r, block, surf, obstacles):
        t = self._plan2(r, block, surf, obstacles)
        o = self._find(t['name'], block, obstacles) if t['kind'] == 'pick' else None
        if o is not None and self.wedges < 5 and not self.no_wedge and tilt(o.th) < 0.2:
            if o.cx < X_MIN - 0.05:
                return {'kind': 'wedge', 'name': o.name, 'side': 'L'}
            if o.cx > X_MAX + 0.05:
                return {'kind': 'wedge', 'name': o.name, 'side': 'R'}
        return t

    def _zone(self, g, bw):
        """region to keep free for placing block at g (plus pusher room if g is beyond reach)"""
        zl, zr = g - bw / 2 - 0.07, g + bw / 2 + 0.07
        if g < X_MIN:
            zr += (X_MIN - g) + 0.42
        elif g > X_MAX:
            zl -= (g - X_MAX) + 0.42
        return zl, zr

    def _plan2(self, r, block, surf, obstacles):
        lo, hi = self._goal_range(block, surf)
        bx = block.cx
        bw = block.aw
        gx_c = float(np.clip(surf.x, lo, hi))
        # --- short push finish when block is close to the goal (e.g. goal beyond reach)
        if abs(bx - gx_c) < 0.15 and tilt(block.th) < 0.2:
            dd = 1.0 if gx_c > bx else -1.0
            pl, pr = min(bx, gx_c) - bw / 2 - 0.03, max(bx, gx_c) + bw / 2 + 0.03
            if not any(o.xr > pl and o.xl < pr for o in obstacles):
                t = self._push_task(block, dd, gx_c, obstacles, protect=None, strict=False)
                if t['reach_ok'] and len(t['chain']) == 1:
                    return t
        # --- option 1: graspable block -> pick and place onto surface
        if bw < GRASP_W and block.ah < 0.6:
            # surface region must be free (with finger clearance)
            best = None
            for g in np.linspace(lo, hi, 21) if hi > lo else [gx_c]:
                zl, zr = self._zone(g, bw)
                if any(o.xr > zl and o.xl < zr for o in obstacles):
                    continue
                if best is None or abs(g - surf.x) < abs(best - surf.x):
                    best = float(g)
            if best is not None:
                return {'kind': 'pick', 'name': block.name, 'place': best}
            zl, zr = self._zone(gx_c, bw)
            blockers = [o for o in obstacles if o.xr > zl and o.xl < zr]
            blockers.sort(key=lambda o: abs(o.cx - gx_c))
            for o in blockers:
                if o.aw < GRASP_W:
                    spot = self._free_spot(o.aw, o.name, block, surf, obstacles,
                                           [(zl - 0.05, zr + 0.05)], o.cx)
                    if spot is not None:
                        return {'kind': 'pick', 'name': o.name, 'place': spot}
            for o in blockers:
                t = self._clear_push(o, zl, zr, block, obstacles)
                if t is not None:
                    return t
            o = blockers[0]
            # relocate the block itself out of the way, then retry
            if self._block_moved < 2:
                far = [(zl - 0.3, zr + 0.3)]
                spots = self._free_spot(block.aw, block.name, block, surf, obstacles, far,
                                        block.cx, all_spots=True)
                for sp in spots[:60]:
                    if abs(sp - block.cx) < 0.05:
                        continue
                    b2 = copy.copy(block)
                    dx = sp - block.cx
                    b2.x += dx; b2.cx += dx; b2.xl += dx; b2.xr += dx
                    if any(self._clear_push(ob, zl, zr, b2, obstacles) is not None
                           for ob in blockers):
                        self._block_moved += 1
                        return {'kind': 'pick', 'name': block.name, 'place': sp}
                if self.wedges < 5 and not self.no_wedge:
                    for o in blockers:
                        if tilt(o.th) > 0.2:
                                continue
                        if o.xl - WALL_L < 0.45:
                                return {'kind': 'wedge', 'name': o.name, 'side': 'L'}
                        if WALL_R - o.xr < 0.45:
                                return {'kind': 'wedge', 'name': o.name, 'side': 'R'}
                if not self._block_moved:
                    spot = self._free_spot(block.aw, block.name, block, surf, obstacles,
                                           [(zl - 0.6, zr + 0.6)], gx_c)
                    if spot is not None and abs(spot - block.cx) > 0.05:
                        self._block_moved += 1
                        return {'kind': 'pick', 'name': block.name, 'place': spot}
            d = 1.0 if o.cx > gx_c else -1.0
            tgt = (zr + o.aw / 2 + 0.02) if d > 0 else (zl - o.aw / 2 - 0.02)
            return self._push_task(o, d, tgt, obstacles + [block])
        # --- option 2: push block
        d = 1.0 if surf.x > bx else -1.0
        cands = [gx_c, lo if d > 0 else hi]
        t0 = self._push_task(block, d, cands[0], obstacles)
        if not t0['reach_ok'] and self.wedges < 5 and not self.no_wedge and tilt(block.th) < 0.2:
            return {'kind': 'wedge', 'name': block.name, 'side': 'L' if d > 0 else 'R'}
        # graspable obstacles in the sweep path: relocate them first
        g1 = cands[0]
        path = (min(bx, g1) - bw / 2 - 0.03, max(bx, g1) + bw / 2 + 0.03)
        inpath = [o for o in obstacles if o.xr > path[0] and o.xl < path[1]
                  and d * (o.cx - bx) > 0 and o.aw < GRASP_W]
        inpath.sort(key=lambda o: d * (o.cx - bx))
        for o in inpath:
            sw = (path[0] - (0.55 if d > 0 else 0.1), path[1] + (0.1 if d > 0 else 0.55))
            spot = self._free_spot(o.aw, o.name, block, surf, obstacles, [sw], o.cx)
            if spot is not None:
                return {'kind': 'pick', 'name': o.name, 'place': spot}
            # no room: bring block up to the obstacle first, then retry
            gpart = (o.xl - bw / 2 - 0.12) if d > 0 else (o.xr + bw / 2 + 0.12)
            if d * (gpart - bx) > 0.1:
                t = self._push_task(block, d, gpart, obstacles)
                if t['reach_ok']:
                    return t
            break
        for g in cands:
            if self._chain_ok(g, d, bw, bx, obstacles):
                return self._push_task(block, d, g, obstacles)
        # infeasible chain: relocate a graspable obstacle ahead
        g = cands[1]
        sweep = (min(bx, g) - bw / 2 - 0.5, max(bx, g) + bw / 2 + 0.05) if d > 0 else \
                (min(bx, g) - bw / 2 - 0.05, max(bx, g) + bw / 2 + 0.5)
        ahead = [o for o in obstacles if d * (o.cx - bx) > 0]
        ahead.sort(key=lambda o: -d * o.cx)   # farthest first (nearest wall)
        for o in ahead:
            if o.aw < GRASP_W:
                spot = self._free_spot(o.aw, o.name, block, surf, obstacles, [sweep], o.cx)
                if spot is not None:
                    return {'kind': 'pick', 'name': o.name, 'place': spot}
        return self._push_task(block, d, g, obstacles)

    def _push_task(self, target, d, gx, others, protect=None, strict=False):
        near = target.xl if d > 0 else target.xr
        chain = [target.name]
        changed = True
        while changed:
            changed = False
            px = near - d * (PUSH_OFF + 0.05)
            zl, zr = px - CLEAR_HALF, px + CLEAR_HALF
            for o in others:
                if o.name in chain or o.name == target.name:
                    continue
                if o.xr > zl and o.xl < zr and d * (o.cx - target.cx) < 0:
                    chain.append(o.name)
                    near = min(near, o.xl) if d > 0 else max(near, o.xr)
                    changed = True
        extra = min(0.3, self.px_extra.get(target.name, 0.0)) if hasattr(self, 'px_extra') else 0.0
        px = near - d * (PUSH_OFF + 0.05 + extra)
        px = float(np.clip(px, X_MIN, X_MAX))
        reach = d * (near - px)
        pgap = float(np.clip(2 * (reach - 0.055), GAP_MIN, GAP_OPEN))
        reach_ok = reach >= 0.105
        if strict:
            if protect is not None and protect in chain:
                return None
            if not reach_ok:
                return None
        return {'kind': 'push', 'name': target.name, 'd': d, 'gx': gx, 'px': px,
                'reach_ok': reach_ok, 'gap': pgap, 'chain': chain}

    # ------------------------------------------------------------------
    def _wall_gap(self, side, block, obstacles):
        g = 10.0
        top = FLOOR
        for o in obstacles + [block]:
            if side == 'L':
                if o.xl < 0.7:
                    g = min(g, o.xl - WALL_L)
                    if o.xl < 0.5:
                        top = max(top, o.yt)
            else:
                if o.xr > WALL_R - 0.7:
                    g = min(g, WALL_R - o.xr)
                    if o.xr > WALL_R - 0.5:
                        top = max(top, o.yt)
        return g, top

    def _wedge(self, r, T, block, obstacles):
        """push objects away from a wall by pressing the round base down next to it.
        returns action, or None if the phase changed (caller loops)"""
        side = T['side']
        sgn = 1.0 if side == 'L' else -1.0
        wall = WALL_L if side == 'L' else WALL_R
        gap, top = self._wall_gap(side, block, obstacles)
        yw = float(min(1.7, max(Y_TRAVEL, top + 0.75)))
        ph = self.phase
        if ph == 'w_rise':
            a, done = self._goto(r, [None, yw, None, ARM_MIN, GAP_OPEN])
            if done:
                self.phase = 'w_x1'
                return None
            return a
        if ph == 'w_x1':
            a, done = self._goto(r, [wall + sgn * 0.5, yw, None, ARM_MIN, GAP_OPEN])
            if done:
                self.phase = 'w_rotup'
                self.rot_target = r[2] + sgn * np.pi
                return None
            return a
        if ph == 'w_rotup' or ph == 'w_rotdown':
            dth = self.rot_target - r[2]
            if abs(dth) < 2e-3:
                if ph == 'w_rotup':
                    self.phase = 'w_x2'
                else:
                    self.phase = 'plan'
                return None
            a = np.zeros(5)
            a[2] = np.clip(dth, -0.19, 0.19)
            return a
        if ph == 'w_x2':
            a, done = self._goto(r, [wall + sgn * X_MIN, yw, None, ARM_MIN, GAP_OPEN])
            if done:
                self.phase = 'w_down'
                self.prev_y = None
                self.stuck = 0
                return None
            return a
        if ph == 'w_down':
            stuck = False
            if self.prev_y is not None and abs(r[1] - self.prev_y) < 1e-4:
                self.stuck += 1
                stuck = self.stuck > 2
            self.prev_y = r[1]
            tgt = self._find(T['name'], block, obstacles)
            tipped = tgt is not None and (tilt(tgt.th) > 0.25 or tgt.yb < FLOOR - 0.10)
            if gap >= 0.38 or stuck or r[1] <= 0.36 or tipped:
                self.phase = 'w_up2'
                return None
            a = np.zeros(5)
            if r[1] - 0.24 > top + 0.04:
                a[1] = -min(0.049, r[1] - 0.24 - top - 0.03)
            else:
                a[1] = -0.005
            return a
        if ph == 'w_up2':
            a, done = self._goto(r, [None, yw, None, ARM_MIN, GAP_OPEN])
            a[0] = 0.0
            if done:
                self.phase = 'w_x3'
                return None
            return a
        if ph == 'w_x3':
            a, done = self._goto(r, [wall + sgn * 0.5, yw, None, ARM_MIN, GAP_OPEN])
            if done:
                self.phase = 'w_rotdown'
                self.rot_target = r[2] - sgn * np.pi
                return None
            return a
        self.phase = 'plan'
        return None

    # ------------------------------------------------------------------
    def _find(self, name, block, obstacles):
        if block.name == name:
            return block
        for o in obstacles:
            if o.name == name:
                return o
        return None

    def get_action(self, state):
        self.t += 1
        r, block, surf, obstacles = self._parse(state)
        a = np.zeros(5)
        for _ in range(8):
            ph = self.phase
            T = self.task
            if ph == 'init':
                xs = float(np.clip(r[0], 0.55, WALL_R - 0.55))
                a, done = self._goto(r, [None, Y_TRAVEL, None, ARM_MIN, GAP_OPEN])
                if r[1] > Y_TRAVEL - 0.01:
                    a, done = self._goto(r, [xs, Y_TRAVEL, None, ARM_MIN, GAP_OPEN])
                if r[1] > Y_TRAVEL - 0.01 and abs(r[0] - xs) < 0.01:
                    a2, done2 = self._goto(r, [xs, Y_TRAVEL, DOWN, ARM_MIN, GAP_OPEN])
                    a = a2
                    if done2:
                        self.phase = 'plan'
                        continue
                return a
            if ph == 'plan':
                self.task = self._plan(r, block, surf, obstacles)
                self.phase = 'move'
                if self.task['kind'] == 'wedge':
                    o = self._find(self.task['name'], block, obstacles)
                    og = o.xl - WALL_L if self.task['side'] == 'L' else WALL_R - o.xr
                    key = (self.task['name'], self.task['side'])
                    if key in self.wedge_gap and og < self.wedge_gap[key] + 0.02:
                        self.no_wedge = True
                        self.task = self._plan(r, block, surf, obstacles)
                    self.wedge_gap[key] = og
                if self.task['kind'] == 'wedge':
                    self.wedges += 1
                    self.phase = 'w_rise'
                continue
            if ph == 'retreat':
                a, done = self._goto(r, [None, Y_TRAVEL, DOWN, ARM_MIN, GAP_OPEN])
                a[0] = 0.0
                if done:
                    self.phase = 'plan'
                    continue
                return a
            obj = self._find(T['name'], block, obstacles)
            if T['kind'] == 'push':
                if ph == 'move':
                    a, done = self._goto(r, [T['px'], Y_TRAVEL, DOWN, ARM_MIN, T.get('gap', GAP_OPEN)])
                    if done:
                        self.phase = 'extend'
                        continue
                    return a
                if ph == 'extend':
                    a, done = self._goto(r, [T['px'], None, DOWN, ARM_MAX, T.get('gap', GAP_OPEN)])
                    if done:
                        self.phase = 'descend'
                        self.stuck = 0
                        self.prev_y = None
                        continue
                    return a
                if ph == 'descend':
                    a, done = self._goto(r, [T['px'], Y_PUSH, DOWN, ARM_MAX, T.get('gap', GAP_OPEN)])
                    if done:
                        self.phase = 'push'
                        continue
                    if self.prev_y is not None and abs(r[1] - self.prev_y) < 1e-4:
                        self.stuck += 1
                        if self.stuck > 2:
                            if r[1] < Y_PUSH + 0.06:
                                self.phase = 'push'
                                continue
                            self.px_extra[T['name']] = self.px_extra.get(T['name'], 0.0) + 0.08
                            self.phase = 'retreat'
                            continue
                    self.prev_y = r[1]
                    return a
                if ph == 'push':
                    d = T['d']
                    err = T['gx'] - obj.cx
                    lost = d * (r[0] - (obj.xl if d > 0 else obj.xr)) > 0.0
                    if d * err <= 0.002 or tilt(obj.th) > 0.3 or lost:
                        self.phase = 'retreat'
                        continue
                    v = np.clip(err, -0.049, 0.049)
                    if abs(err) < 0.1:
                        v = np.clip(err, -0.02, 0.02)
                    a = np.zeros(5)
                    a[0] = v
                    a[1] = np.clip(Y_PUSH - r[1], -0.049, 0.0)
                    return a
            if T['kind'] == 'wedge':
                a = self._wedge(r, T, block, obstacles)
                if a is None:
                    continue
                return a
            if T['kind'] == 'pick':
                if ph == 'move':
                    a, done = self._goto(r, [float(np.clip(obj.cx, X_MIN, X_MAX)), Y_TRAVEL, DOWN, ARM_MIN, GAP_OPEN])
                    if done:
                        self.phase = 'extend'
                        continue
                    return a
                if ph == 'extend':
                    a, done = self._goto(r, [float(np.clip(obj.cx, X_MIN, X_MAX)), None, DOWN, ARM_MAX, GAP_OPEN])
                    if done:
                        self.phase = 'descend'
                        self.stuck = 0
                        self.prev_y = None
                        continue
                    return a
                if ph == 'descend':
                    yt = max(Y_PUSH, obj.yb + 0.015 + FTIP + ARM_MAX, obj.yt + 0.525)
                    a, done = self._goto(r, [float(np.clip(obj.cx, X_MIN, X_MAX)), yt, DOWN, ARM_MAX, GAP_OPEN])
                    if done:
                        self.phase = 'close'
                        continue
                    if self.prev_y is not None and abs(r[1] - self.prev_y) < 1e-4:
                        self.stuck += 1
                        if self.stuck > 2:
                            self.phase = 'close'
                            continue
                    self.prev_y = r[1]
                    return a
                if ph == 'close':
                    if obj.held:
                        self.grasp_off = r[1] - obj.y
                        self.phase = 'lift'
                        continue
                    if r[4] <= GAP_MIN + 0.005:
                        self.fails += 1
                        self.phase = 'retreat'
                        continue
                    a = np.zeros(5)
                    a[4] = -0.0199
                    return a
                if ph == 'lift':
                    if not obj.held:
                        self.phase = 'retreat'
                        continue
                    yc = min(1.7, 0.75 + obj.ah / 2 + self.grasp_off)
                    a, done = self._goto(r, [None, yc, DOWN, ARM_MAX, None], tol=5e-3)
                    if done:
                        self.phase = 'carry'
                        continue
                    return a
                if ph == 'carry':
                    if not obj.held:
                        self.phase = 'retreat'
                        continue
                    a, done = self._goto(r, [float(np.clip(T['place'], X_MIN, X_MAX)), None, DOWN, ARM_MAX, None], tol=2e-3)
                    if done:
                        self.phase = 'lower'
                        self.stuck = 0
                        self.prev_y = None
                        continue
                    return a
                if ph == 'lower':
                    yt = FLOOR + obj.ah / 2 + 0.008 + self.grasp_off
                    a, done = self._goto(r, [float(np.clip(T['place'], X_MIN, X_MAX)), yt, DOWN, ARM_MAX, None], tol=2e-3)
                    stuck = False
                    if self.prev_y is not None and abs(r[1] - self.prev_y) < 1e-4:
                        self.stuck += 1
                        stuck = self.stuck > 2
                    self.prev_y = r[1]
                    if done or stuck:
                        self.phase = 'release'
                        self.release_gap = None
                        continue
                    return a
                if ph == 'release':
                    if self.release_gap is None:
                        self.release_gap = min(GAP_OPEN, r[4] + 0.04)
                    a, done = self._goto(r, [None, None, DOWN, ARM_MAX, self.release_gap])
                    if done:
                        self.phase = 'retreat'
                        continue
                    return a
            # unknown -> replan
            self.phase = 'retreat'
        return a
