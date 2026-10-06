

CARRY_ARM = np.array([0.67717022, -0.34313199, 1.2, -1.46688402, 1.24223232, -1.95442832, 2.22254133])
SIDE_YAWS = (0.0, np.pi, np.pi / 2, -np.pi / 2)
RING = [(-0.85, -1.15), (-0.85, 0.0), (-0.85, 1.15), (0.0, 1.15), (0.85, 1.15), (0.85, 0.0),
        (0.85, -1.15), (0.0, -1.15)]


_OCC_B64 = "eNrNWN1t2zAQJs0gzEMBZYAA8iSRB+gQGSEDBKA26Ep87BZl0QX4Fj0QYqU7lUdSki25kBQ+WKDMM7/7/86MrVtn0X9y3/aPwjf9Q3nffUrvXfeovDf9AXhXeDjRHdCMCQ9Spfe2e3QH6ltCNQgZFGIg1OBNGoRagAI/V8DpTqj/ORIyC29aAU/iAQGfe6wTe2MAQpPlKwDO0RAlPhBygZAVoBQIuQI9eXLAg04jpVHBBoU0CDm0lAmeFnggXNuSVSq4goQsCuFN/VclHsfb5QS8MuAnn8jgY4i6GqLuiW9u+eGGF1EDiBaD3SIiCjcZ9NRBJwVqCNSzDO/IJ1dN3ibWwzRwSUSvExpuWgIvjfIKTa5kv+G27DeSvbNd1+ntOya/CREm4yAmnYYw0mCIFk/bOPLbJB7ROE1snCkhd6/QTSj0To+KzS8IrQbs/yp61dnliR23KMqbOJvJDUXsBoE6Ubpb9FMNCjpUehuhBoX0WIi8f7sBYKk7aj2fGBRaKDnlv0bnCDJlMxqnzbMZDVEEQ+zuBhu7IYfHQ+ZSyDP1icXmB3TWpugrLKuf97X8u8Rgd1TsC2x0Cc1RYwV5ovRMzd+u2KhxLRzBa2J4cTaU0NSsgnDjlgyx4zo/UHfnsckpS0lBssqUpcxENicZfheZMXEQNHl998HTeWEkeG4CSlSAfsPmgdWH1BzVbkfmq+WUfYbnF8vHhvVkniEUgVn/yM77WJzjRdjdebCAziHPkPmZ2CsSrvj/UZ4QzMSiU4RqhsybnMxL9DQOMgw3Ejfftk+AgcN+SE11T8Xw1HK2TH66l56vEJLTxX4RmQ/Jldgf4t+8oF0uO0yzF+ru5R+cn5SdDvk5tpzH3ja9dEroOhQb86BkQHEh04Y55vHy80uQ+ZvFkgpLQv4Smnkg20+EZgrj1yDz5pWFkB/Gu6RnFYEZW9zU2Thz4DBlMpKbwBN5PlELYUl/PYrM6w/6E0M6peNm5RKd9ISeh7J4l88Vo1kjH1l5IE84Ox5I5oW3CSUb0bQpMj/6U9CMiYZbzWVyAjSCkrLIq3NFROZ9VkB1GBqHdL+fzP8FbzCkBw=="
_OX0, _OY0, _ORES, _ONX, _ONY, _ONT = -1.3, -1.4, 0.05, 53, 57, 16


def _load_occ():
    import base64, zlib
    bits = np.frombuffer(zlib.decompress(base64.b64decode(_OCC_B64)), np.uint8)
    return np.unpackbits(bits)[:_ONT * _ONY * _ONX].reshape(_ONT, _ONY, _ONX).astype(bool)


OCC = _load_occ()


def occ_hit(x, y, yaw):
    """Conservative table-collision test for base poses with CARRY_ARM (vectorised)."""
    x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float))
    yaw = np.broadcast_to(np.asarray(yaw, float), x.shape)
    fi = (x - _OX0) / _ORES; fj = (y - _OY0) / _ORES
    ft = ((yaw + np.pi) % (2 * np.pi)) / (2 * np.pi / _ONT)
    hit = np.zeros(x.shape, bool)
    inside = (fi >= 0) & (fi < _ONX - 1) & (fj >= 0) & (fj < _ONY - 1)
    i0 = np.clip(np.floor(fi).astype(int), 0, _ONX - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, _ONY - 2)
    t0 = np.floor(ft).astype(int) % _ONT
    for dt in (0, 1):
        t = (t0 + dt) % _ONT
        for di in (0, 1):
            for dj in (0, 1):
                hit |= OCC[t, j0 + dj, i0 + di]
    return hit & inside


def _mk_wp():
    pts = []
    ys = np.round(np.arange(-1.15, 1.151, 0.1), 3)
    xs = np.round(np.arange(-0.85, 0.851, 0.1), 3)
    for ax in (0.65, 0.75, 0.85):
        for sx in (-1, 1):
            for y in ys:
                pts.append((sx * ax, y))
    for by in (0.95, 1.05, 1.15):
        for sy in (-1, 1):
            for x in xs:
                pts.append((x, sy * by))
    return np.unique(np.array(pts), axis=0)


WP_XY = _mk_wp()


def _vsteps(A, B):
    d = B[None, :, :] - A[:, None, :]
    for i in (2, 7, 9):
        if i >= d.shape[2]:
            break
        d[:, :, i] = (d[:, :, i] + np.pi) % (2 * np.pi) - np.pi
    m = np.max(np.abs(d), axis=2)
    return np.maximum(1, np.ceil(m / STEP - 1e-6)).astype(int)


def seg_hit(a, b, n, include_end):
    """Grid test along the base path a->b (n steps); intermediate samples (+ end)."""
    m = n + 1 if include_end else n
    if m <= 1:
        return False
    ts = np.arange(1, m) / n
    d = b[:3] - a[:3]
    d[2] = wrap(d[2])
    pts = a[:3][None] + ts[:, None] * d[None]
    return bool(np.any(occ_hit(pts[:, 0], pts[:, 1], pts[:, 2])))


def seg_hit_many(A, B, N, relax_a=False, relax_b=False, rr=0.25):
    """Vectorised seg_hit (intermediate samples only) for rows A[i]->B[i] with N[i] steps."""
    A = np.asarray(A, float)[:, :3]; B = np.asarray(B, float)[:, :3]
    N = np.asarray(N, int)
    n = len(A)
    if n == 0:
        return np.zeros(0, bool)
    S = max(1, int(N.max()) - 1)
    k = np.arange(1, S + 1)[None, :]
    valid = k < N[:, None]
    ts = np.where(valid, k / np.maximum(N[:, None], 1), 0.0)
    d = B - A
    d[:, 2] = (d[:, 2] + np.pi) % (2 * np.pi) - np.pi
    pts = A[:, None, :] + ts[:, :, None] * d[:, None, :]
    h = occ_hit(pts[..., 0].ravel(), pts[..., 1].ravel(), pts[..., 2].ravel()).reshape(n, S)
    if relax_a:
        valid = valid & (np.hypot(pts[..., 0] - A[:, None, 0], pts[..., 1] - A[:, None, 1]) > rr)
    if relax_b:
        valid = valid & (np.hypot(pts[..., 0] - B[:, None, 0], pts[..., 1] - B[:, None, 1]) > rr)
    return np.any(h & valid, axis=1)


def side_normal(th0):
    return np.array([-np.cos(th0), -np.sin(th0)])


def base_stations(tx, ty):
    out = []
    for th0 in SIDE_YAWS:
        for dth in (0.0, -0.4, 0.4):
            th = th0 + dth
            c, s_ = np.cos(th), np.sin(th)
            fx, ly = 0.5, 0.19
            bx = tx - (c * fx - s_ * ly)
            by = ty - (s_ * fx + c * ly)
            if th0 == 0.0:
                bx = min(bx, -0.47)
            elif th0 == np.pi:
                bx = max(bx, 0.47)
            elif th0 > 0:
                by = min(by, -0.97)
            else:
                by = max(by, 0.97)
            out.append((np.array([bx, by, wrap(th)]), th0))
    return out


class Planner:
    def __init__(self, sim):
        self.sim = sim
        self.nsim = 0
        self._wpcache = {}

    # ---- low level simulation ----
    def simulate_exact(self, flat, actions):
        s = self.sim
        s.set_state(flat)
        cur = np.array(flat, float)
        term = False
        for a in actions:
            exp = cur[:10] + a[:10]
            o, _, term, _, _ = s.env.step(np.asarray(a, np.float32))
            o = np.array(o, float)
            if np.max(np.abs(conf_delta(o[:10], exp))) > 1e-3:
                return False, o, term
            cur = o
        return True, cur, term

    def _apply(self, q, a):
        t = q + np.asarray(a[:10], np.float32).astype(float)
        t = np.clip(t, self.sim.lo, self.sim.hi)
        for d in (2, 7, 9):
            t[d] = wrap(t[d])
        return t

    def simulate(self, flat, actions):
        """Fast: collision-check intermediate waypoints, exact env.step for the last action."""
        s = self.sim
        cid = s.cid
        k = s.env._num_blocks if hasattr(s.env, '_num_blocks') else len(s.env.blocks)
        s.set_state(flat)
        blocks = list(s.env.blocks)
        held = None
        for i in range(len(blocks)):
            if flat[19 + 20 + 11 * i + 7] > 0.5:
                held = i
        obst = [s.env.table, s.env.plate] + [b for i, b in enumerate(blocks) if i != held]
        G = None
        if held is not None and flat[11] > 0.5:
            G = _T(flat[12:15], flat[15:19])
        q = np.array(flat[:10], float)
        for si_, a in enumerate(actions[:-1]):
            self.nsim += 1
            q = self._apply(q, a)
            s.set_conf(q)
            for o in obst:
                cp = p.getClosestPoints(s.robot, o, 0.0, physicsClientId=cid)
                if cp:
                    self.last_fail = (si_, o, cp[0][3], np.round(q[:3], 2))
                    return False, None, False
            if G is not None:
                B = s.ff.fk(q) @ G
                p.resetBasePositionAndOrientation(blocks[held], B[:3, 3], mat_to_quat(B[:3, :3]), physicsClientId=cid)
                for o in obst:
                    if p.getClosestPoints(blocks[held], o, 0.0, physicsClientId=cid):
                        return False, None, False
        if len(actions) > 1:
            vec = np.array(flat, float)
            vec[:10] = q
            if G is not None:
                B = s.ff.fk(q) @ G
                o_ = 19 + 20 + 11 * held
                vec[o_:o_ + 3] = B[:3, 3]
                vec[o_ + 3:o_ + 7] = mat_to_quat(B[:3, :3])
        else:
            vec = flat
        return self.simulate_exact(vec, actions[-1:])

    def path_actions(self, qa, qb, n, grip_last=0.0):
        d = conf_delta(qa, qb) / n
        acts = []
        for i in range(n):
            a = np.zeros(11)
            a[:10] = d
            if i == n - 1:
                a[10] = grip_last
            acts.append(a)
        return acts

    def seq_actions(self, qc, seq, grip_last, extra=0):
        acts = []
        prev = qc
        for i, w in enumerate(seq):
            n = nsteps(prev, w) + (extra if i == len(seq) - 1 else 0)
            acts += self.path_actions(prev, w, n, grip_last if i == len(seq) - 1 else 0.0)
            prev = w
        return acts

    def seq_cost(self, qc, seq):
        c = 0
        prev = qc
        for w in seq:
            c += nsteps(prev, w)
            prev = w
        return c

    def table_collision(self, q):
        s = self.sim
        s.set_conf(q)
        with s.env.client():
            from pybullet_tools.utils import pairwise_collision
            return pairwise_collision(s.robot, s.env.table) or pairwise_collision(s.robot, s.env.plate)

    def _ik_pushout(self, pos, quat, seed, normal, use_base, wbase=1.5):
        """IK; if robot collides with table, push base outward along normal and retry arm-only."""
        s = self.sim
        dofs = ([0, 1, 2] if use_base else []) + list(range(3, 10))
        w = ([wbase] * 3 if use_base else []) + [1.0] * 7
        q = s.ik(pos, quat, seed, dofs, weights=w)
        if q is None:
            return None
        for attempt in range(6):
            if not self.table_collision(q):
                return q
            if normal is None:
                # outward from table center
                v = np.array([q[0], q[1]])
                if abs(v[0]) / 0.3 > abs(v[1]) / 0.6:
                    normal = np.array([np.sign(v[0]), 0.0])
                else:
                    normal = np.array([0.0, np.sign(v[1])])
            q2 = q.copy()
            q2[:2] += 0.05 * normal
            q = s.ik(pos, quat, q2, list(range(3, 10)))
            if q is None:
                return None
        return None

    def ik_candidates(self, pos, quat, qc, use_stations=True, best_cost=1e9):
        sols = []
        q = self._ik_pushout(pos, quat, qc, None, True)
        if q is not None:
            sols.append(q)
        if use_stations:
            for st, th0 in base_stations(pos[0], pos[1]):
                seed = np.concatenate([st, qc[3:]])
                if nsteps(qc, seed[:3].tolist() + list(qc[3:])) >= best_cost:
                    continue
                q = self._ik_pushout(pos, quat, seed, side_normal(th0), False)
                if q is None:
                    seed2 = np.concatenate([st, CARRY_ARM])
                    q = self._ik_pushout(pos, quat, seed2, side_normal(th0), False)
                if q is not None:
                    sols.append(q)
        return sols

    def path_seqs(self, qc, qt, pre=None, topn=60):
        """Candidate waypoint sequences from qc to qt (sorted by cost)."""
        seqs = [[qt]]
        if pre is not None:
            seqs.append([pre, qt])
        lift = qc.copy()
        lift[3:] = CARRY_ARM
        seqs.append([lift, qt])
        if pre is not None:
            seqs.append([lift, pre, qt])
        costed = [(self.seq_cost(qc, sq), i, sq) for i, sq in enumerate(seqs)]
        tail = [pre, qt] if pre is not None else [qt]
        first = tail[0]
        tail_cost = self.seq_cost(first, tail[1:]) if len(tail) > 1 else 0
        yc, yt = qc[2], qt[2]
        dy = wrap(yt - yc)
        yaws = [yc, wrap(yc + dy / 2), yt]
        P = WP_XY
        W = []
        for yv in yaws:
            B = np.zeros((len(P), 10))
            B[:, :2] = P
            B[:, 2] = yv
            B[:, 3:] = CARRY_ARM
            W.append(B)
        W = np.vstack(W)
        W = W[~occ_hit(W[:, 0], W[:, 1], W[:, 2])]
        c1 = _vsteps(qc[None], W)[0]
        c2 = _vsteps(W, first[None])[:, 0]
        single = c1 + c2 + tail_cost
        nW = len(W)
        bad1 = seg_hit_many(np.repeat(qc[None], nW, 0), W, c1, relax_a=True)
        bad2 = seg_hit_many(W, np.repeat(first[None], nW, 0), c2, relax_b=True)
        okW = ~(bad1 | bad2)
        sc = np.where(okW, single, 10 ** 6)
        idx = np.argsort(sc, kind='stable')[:topn]
        n0 = len(seqs)
        for j, i in enumerate(idx):
            if sc[i] >= 10 ** 6:
                break
            costed.append((int(single[i]), n0 + j, [W[i]] + tail))
        # pairs among the most promising first / second waypoints
        k1 = np.where(bad1, 10 ** 6, c1 + _vsteps(W, qt[None])[:, 0])
        k2 = np.where(bad2, 10 ** 6, c2 + _vsteps(qc[None], W)[0])
        a1 = np.argsort(k1, kind='stable')[:100]
        a2 = np.argsort(k2, kind='stable')[:100]
        a1 = a1[k1[a1] < 10 ** 6]
        a2 = a2[k2[a2] < 10 ** 6]
        if len(a1) and len(a2):
            M = _vsteps(W[a1][:, :3], W[a2][:, :3])
            pair = c1[a1][:, None] + M + c2[a2][None, :] + tail_cost
            pair[a1[:, None] == a2[None, :]] = 10 ** 6
            order = np.argsort(pair, axis=None, kind='stable')[:600]
            i1s, i2s = np.unravel_index(order, pair.shape)
            keep = pair[i1s, i2s] < 10 ** 6
            i1s, i2s = i1s[keep], i2s[keep]
            bad = seg_hit_many(W[a1[i1s]], W[a2[i2s]], M[i1s, i2s])
            n1 = n0 + len(idx)
            got = 0
            for j in np.nonzero(~bad)[0]:
                i1, i2 = i1s[j], i2s[j]
                costed.append((int(pair[i1, i2]), n1 + got, [W[a1[i1]], W[a2[i2]]] + tail))
                got += 1
                if got >= topn:
                    break
        costed.sort(key=lambda c: (c[0], c[1]))
        return costed

    def wp_ok(self, w):
        key = tuple(np.round(w, 3))
        r = self._wpcache.get(key)
        if r is None:
            r = not self.table_collision(w)
            self._wpcache[key] = r
        return r

    def best_path(self, flat, qc, qt, grip_last, check, pre=None, best_cost=1e9, max_tries=40):
        tries = 0
        for cost, _, seq in self.path_seqs(qc, qt, pre):
            if cost >= best_cost:
                break
            for extra in (0, 1):
                if cost + extra >= best_cost:
                    break
                if extra == 0 and not all(self.wp_ok(w) for w in seq[:-1]):
                    break
                acts = self.seq_actions(qc, seq, grip_last, extra)
                ok, fin, term = self.simulate(flat, acts)
                tries += 1
                if ok and check(fin):
                    return acts, fin, term
            if tries > max_tries:
                break
        return None
