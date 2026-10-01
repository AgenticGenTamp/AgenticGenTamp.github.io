"""Plan for PR2Blocked: move blocker aside, grasp penned green block, drop on plate."""
import numpy as np
from basepath import base_route
from collide import arm_hits
from kin import ik, fk_world, fk_points, wrap, rect_pen

TABLE = (4.5, 0.0, 0.3, 0.6)   # near table cx, cy, hx, hy
PLATE = (4.5, -0.3, 0.3, 0.3)
HIGH_Q = np.array([0.39, 0.33, 0.0, -1.52, 2.72, -1.22, -2.99])
ELBOW_MIN = 0.88
GRASP_DZ = 0.035
CART_STEP = 0.02
GRASP_BACK = 0.025
DEBUG = False
PHI_TRIES = 5
W_NAT = 1.0
W_ROLL = 0.5


def side_seeds(table, tgt):
    cx, cy, hx, hy = table
    m = 0.40
    out = []
    # -x side
    for yaw in (0.0, 0.6, -0.6, np.pi / 2, -np.pi / 2):
        out.append(np.array([cx - hx - m, tgt[1] - 0.2 * np.cos(yaw), yaw]))
    # +y side
    for yaw in (-np.pi / 2, np.pi, 0.0, -2.2, -0.9):
        out.append(np.array([tgt[0] + 0.2 * np.sin(yaw + np.pi / 2), cy + hy + m, yaw]))
    # -y side
    for yaw in (np.pi / 2, 0.0, np.pi, 0.9, 2.2):
        out.append(np.array([tgt[0] - 0.2 * np.sin(yaw - np.pi / 2), cy - hy - m, yaw]))
    # +x side (only if reachable within base limit)
    if cx + hx + m < 4.97:
        out.append(np.array([cx + hx + m, tgt[1] + 0.2, np.pi]))
    return out


def valid_spot(p, table, plate, avoid, r=0.055, avoid_r=0.16):
    if abs(p[0] - table[0]) > table[2] - r or abs(p[1] - table[1]) > table[3] - r:
        return False
    inx = abs(p[0] - plate[0]) - plate[2]
    iny = abs(p[1] - plate[1]) - plate[3]
    fully_in = inx < -r and iny < -r
    fully_out = inx > r or iny > r
    if not (fully_in or fully_out):
        return False
    for c in avoid:
        if np.linalg.norm(np.asarray(p[:2]) - np.asarray(c[:2])) < avoid_r:
            return False
    return True


def jdist(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    d[4] = wrap(d[4]); d[6] = wrap(d[6])
    return d


def _rot2(v, ang):
    c, s_ = np.cos(ang), np.sin(ang)
    return np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1], v[2] if len(v) > 2 else 0.0])


FREE_BASE_CHAIN = True
W_BASE_CHAIN = 0.05


def cart_path(base, q0, p0, p1, d, step=CART_STEP, d1=None, table=TABLE):
    """Dense joint path moving tool linearly from p0 to p1 (approach d -> d1). None if IK fails."""
    if d1 is None:
        d1 = d
    dang = wrap(np.arctan2(d1[1], d1[0]) - np.arctan2(d[1], d[0]))
    n = max(1, int(np.ceil(max(np.linalg.norm(p1 - p0) / step, abs(dang) / 0.15))))
    qs = []
    q = q0
    for i in range(1, n + 1):
        t = i / n
        p = p0 + (p1 - p0) * t
        dd = _rot2(d, dang * t)
        if FREE_BASE_CHAIN:
            nb, q, err = ik(p, dd, base, q, free_base=True, base_nom=base, w_base=W_BASE_CHAIN,
                            elbow_min=ELBOW_MIN, table=table, n_restarts=0 if i < n else 3)
            if rect_pen(nb, 0.35, table) > 0:
                return None
        else:
            nb = base
            _, q, err = ik(p, dd, base, q, free_base=False, elbow_min=ELBOW_MIN, n_restarts=0 if i < n else 3)
        if err > 5e-3:
            return None
        if arm_hits(nb, q) is not None:
            if DEBUG: print('    hit', arm_hits(nb, q))
            return None
        if np.abs(jdist(qs[-1][1] if qs else q0, q)).max() > 0.5:
            return None  # branch jump
        base = nb
        qs.append((nb, q))
    return qs


def route_len(a, b, table):
    pts = [np.asarray(a, float)] + base_route(a, b, table)
    L = sum(np.linalg.norm(pts[i + 1][:2] - pts[i][:2]) for i in range(len(pts) - 1))
    return L + 0.3 * abs(wrap(b[2] - a[2]))


TRAVEL_R = {'upper': 0.105, 'fore': 0.09, 'grip': 0.065}


def _travel_ok(b, q, table, ignore=()):
    from kin import fk_points
    import collide
    cx, cy, hx, hy = table
    pts = fk_points(b, q)
    samp = [pts[1], pts[2], pts[3], 0.5 * (pts[0] + pts[1]), 0.5 * (pts[1] + pts[2])]
    for p in samp:
        if abs(p[0] - cx) < hx + 0.12 and abs(p[1] - cy) < hy + 0.12 and p[2] < 0.80:
            if TRAVEL_DEBUG: print('     zfail', np.round(p, 2))
            return False
    sk = collide.CTX['skip']
    rl = dict(collide.R_LINK)
    collide.CTX['skip'] = set(ignore)
    collide.R_LINK.update(TRAVEL_R)
    try:
        h = arm_hits(b, q)
        if TRAVEL_DEBUG and h is not None: print('     hit', h)
        return h is None
    finally:
        collide.CTX['skip'] = sk
        collide.R_LINK.update(rl)


def _count_bad(cfgs, b_from, q_from, b_to, q_to, table, win=0.05, ignore=()):
    t0 = fk_world(b_from, q_from)[0]; t1 = fk_world(b_to, q_to)[0]
    bad = 0
    for bb, qq in cfgs:
        t = fk_world(bb, qq)[0]
        if np.linalg.norm(t - t0) < win or np.linalg.norm(t - t1) < win:
            continue
        if not _travel_ok(bb, qq, table, ignore):
            bad += 1
    return bad


SPREAD_ARM = True
TRAVEL_DEBUG = False
Q_TUCK = np.array([0.0, -0.5, 0.0, -2.0, 0.0, -0.1, 0.0])


TRAVEL_MODE = 'spread'   # 'spread' | 'auto' (checked) | 'legacy' | 'tuck'


def travel_configs(b_from, q_from, b_to, q_to, table, step=0.1, ignore=(), mode=None):
    """Dense (base,q) configs. Tries (1) arm interpolated over the whole drive, (2) arm fixed
    then moved on the final leg, (3) tuck arm, drive, extend; first whose intermediate configs
    pass the clearance check is used (else (3))."""
    route = base_route(b_from, b_to, table)
    b_from = np.asarray(b_from, float); b_to = np.asarray(b_to, float)
    q_from = np.asarray(q_from, float); q_to = np.asarray(q_to, float)
    bases = []; lastleg = []
    cur = b_from
    for k, wp in enumerate(route):
        last = k == len(route) - 1
        n = max(1, int(np.ceil(max(np.linalg.norm(wp[:2] - cur[:2]), abs(wrap(wp[2] - cur[2])) * 0.5) / step)))
        for i in range(1, n + 1):
            t = i / n
            bases.append(np.r_[cur[:2] + t * (wp[:2] - cur[:2]), cur[2] + t * wrap(wp[2] - cur[2])])
            lastleg.append((last, t))
        cur = np.asarray(wp, float)
    dq = jdist(q_from, q_to)

    def finish(cfgs):
        if not cfgs or np.abs(jdist(cfgs[-1][1], q_to)).max() > 1e-6 or np.abs(cfgs[-1][0] - b_to).max() > 1e-6:
            cfgs.append((b_to, q_to))
        return cfgs
    mode = TRAVEL_MODE if mode is None else mode
    if not SPREAD_ARM and mode == 'spread':
        mode = 'legacy'
    opts = []
    N = len(bases)
    if mode in ('spread', 'auto') and N > 1:
        sp = finish([(bb, q_from + ((i + 1) / N) * dq) for i, bb in enumerate(bases)])
        if mode == 'spread':
            return sp
        opts.append(sp)
    if mode == 'tuck':
        opts = []
    else:
        opts.append(finish([(bb, q_from + t * dq if last else q_from) for bb, (last, t) in zip(bases, lastleg)]))
        if mode == 'legacy':
            return opts[-1]
    best = None
    for o in opts:
        nb = _count_bad(o, b_from, q_from, b_to, q_to, table, ignore=ignore)
        if TRAVEL_DEBUG: print("  travel", np.round(b_from, 2), "->", np.round(b_to, 2), "bad", nb, "len", len(o), "ign", ignore)
        if nb == 0:
            return o
        if best is None or nb < best[0]:
            best = (nb, o)
    # tuck: keep continuous joints (4, 6) as they are to avoid long rolls
    qt = Q_TUCK.copy(); qt[4] = q_from[4]; qt[6] = q_from[6]
    d1 = jdist(q_from, qt)
    n1 = max(1, int(np.ceil(np.abs(d1).max() / 0.1)))
    cfgs = [(b_from, q_from + (i / n1) * d1) for i in range(1, n1 + 1)]
    d2 = jdist(qt, q_to)
    for i, bb in enumerate(bases):
        cfgs.append((bb, qt + ((i + 1) / N) * d2 if N > 0 else qt))
    cfgs = finish(cfgs)
    if best is None:
        return cfgs
    nb = _count_bad(cfgs, b_from, q_from, b_to, q_to, table, ignore=ignore)
    if nb < best[0]:
        return cfgs
    return best[1]


def make_plan(robot_base, robot_q, blk, g0, block_z, table=TABLE, plate=PLATE):
    d = np.r_[g0[:2] - blk[:2], 0.0]
    d /= np.linalg.norm(d)
    perp = np.array([-d[1], d[0], 0.0])
    z = block_z + GRASP_DZ
    cb = np.array([blk[0], blk[1], z])
    cg = np.array([g0[0], g0[1], z])
    up = np.array([0, 0, 1.0])
    cands = []
    mid = 0.5 * (cg + cb)
    for b0 in side_seeds(table, mid):
        b, q, err = ik(mid, d, b0, HIGH_Q, free_base=True, base_nom=b0, elbow_min=ELBOW_MIN, table=table)
        if DEBUG: print('base seed', b0.round(2), '->', b.round(3), 'err', round(err, 4), 'pen', rect_pen(b, 0.36, table))
        if err > 5e-3 or rect_pen(b, 0.36, table) > 0:
            continue
        sh = fk_points(b, q)[0]
        nat = mid[:2] - sh[:2]
        ang = abs(wrap(np.arctan2(nat[1], nat[0]) - np.arctan2(d[1], d[0])))
        cost = route_len(robot_base, b, table) + W_NAT * ang + W_ROLL * abs(q[2])
        if DEBUG: print('  cand', b.round(2), 'route', round(route_len(robot_base, b, table), 2), 'ang', round(ang, 2), 'roll', round(q[2], 2))
        cands.append((cost, b, q))
    cands.sort(key=lambda c: c[0])
    for _, b, q in cands:
        res = _chain_for_base(b, q, cb, cg, d, perp, z, table, plate, up)
        if res is None:
            continue
        segs = res
        # prepend travel
        tc = travel_configs(robot_base, np.asarray(robot_q, float), b, segs[0]['configs'][-1][1], table)
        segs[0]['configs'] = tc
        return segs
    return None


def drop_candidates(plate, avoid, z, near, margin=0.09, offset=None):
    """Tool targets such that the held block centre (tool + offset) lands well inside the plate."""
    off = np.zeros(3) if offset is None else np.r_[offset[0], offset[1], 0.0]
    pts = []
    for fx in np.linspace(-1, 1, 7):
        for fy in np.linspace(-1, 1, 7):
            c = np.array([plate[0] + fx * (plate[2] - margin), plate[1] + fy * (plate[3] - margin), z])
            if all(np.linalg.norm(c[:2] - a[:2]) > 0.13 for a in avoid):
                pts.append(c - off)
    pts.sort(key=lambda p: np.linalg.norm(p[:2] - near[:2]))
    return pts


def _solve_seq(b, qg, seq):
    segs = []
    _, q, err = ik(seq[0][1], seq[0][3], b, qg, free_base=False, elbow_min=ELBOW_MIN)
    if err > 5e-3:
        if DEBUG: print('  prehi fail')
        return None
    segs.append(dict(name=seq[0][0], configs=[(b, q)], grip=seq[0][2]))
    p = seq[0][1]; dp = seq[0][3]
    for name, tgt, g, dd in seq[1:]:
        qs = cart_path(b, q, p, tgt, dp, d1=dd)
        if qs is None:
            if DEBUG: print('  chain fail at', name)
            return None
        segs.append(dict(name=name, configs=qs, grip=g))
        b, q = qs[-1]
        p = tgt; dp = dd
    return segs


def _chain_for_base(b, qg, cb, cg, d, perp, z, table, plate, up):
    sh = fk_points(b, qg)[0]
    s = -np.sign((sh - cb) @ perp) or 1.0
    aside = None
    for ss in (s, -s):
        for dist in (0.2, 0.24, 0.16, 0.28, 0.13):
            for back in (0.0, 0.05):
                a = cb + dist * ss * perp - back * d
                if valid_spot(a, table, plate, [cg]):
                    aside = a
                    break
            if aside is not None:
                break
        if aside is not None:
            break
    if aside is None:
        if DEBUG: print('  no aside spot')
        return None
    lift = z + 0.16
    segs = None
    nat = cb[:2] - sh[:2]
    nat_ang = np.arctan2(nat[1], nat[0])
    d_ang = np.arctan2(d[1], d[0])
    phis = [0.0, 0.5, -0.5, 0.9, -0.9, np.pi / 2, -np.pi / 2, 1.25, -1.25]
    phis.sort(key=lambda ph: abs(wrap(d_ang + ph - nat_ang)))
    for phi in phis[:PHI_TRIES]:
        db = _rot2(d, phi)
        seq = [
            ("pre_hi", cb - 0.10 * db + 0.10 * up, 0.0, db),
            ("pre", cb - 0.10 * db, 0.0, db),
            ("grasp_b", cb, -1.0, db),
            ("aside", aside + 0.003 * up, 1.0, db),
            ("retract", aside - 0.10 * db, 0.0, db),
            ("axis", cb - 0.03 * d, 0.0, d),
            ("grasp_g", cg - GRASP_BACK * d, -1.0, d),
            ("out", cg + (lift - z) * up, 0.0, d),
        ]
        segs = _solve_seq(b, qg, seq)
        if segs is not None:
            break
    if segs is None:
        return None
    b, q = segs[-1]['configs'][-1]
    p = seq[-1][1]
    # drop: same base first
    drops = drop_candidates(plate, [aside], lift, cg)
    for dp in drops[:4]:
        qs = cart_path(b, q, p, dp, d, step=0.04)
        if qs is not None:
            segs.append(dict(name="drop", configs=qs, grip=1.0))
            return segs
    # lift higher for carrying
    qs = cart_path(b, q, p, p + np.array([0, 0, 0.1]), d, step=0.04)
    if qs is not None:
        segs.append(dict(name="carry_lift", configs=qs, grip=0.0))
        b, q = qs[-1]
    # need a new base for the drop
    best = None
    for dp in drops[:6]:
        for b0 in side_seeds(table, dp):
            b2, q2, err = ik(dp, d, b0, q, free_base=True, base_nom=b0, elbow_min=ELBOW_MIN, table=table)
            if err > 5e-3 or rect_pen(b2, 0.36, table) > 0:
                continue
            L = route_len(b, b2, table)
            if best is None or L < best[0]:
                best = (L, b2, q2)
        if best is not None:
            break
    if best is None:
        if DEBUG: print('  no drop base')
        return None
    _, b2, q2 = best
    segs.append(dict(name="carry", configs=travel_configs(b, q, b2, q2, table), grip=1.0))
    return segs
