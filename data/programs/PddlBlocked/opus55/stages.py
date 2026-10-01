"""Staged, option-enumerating planners used by the runtime executor."""
import numpy as np
from collide import arm_hits
from kin import ik, fk_points, fk_world, wrap, rect_pen
from planner import (jdist, side_seeds, valid_spot, cart_path, route_len, travel_configs, drop_candidates,
                     _rot2, HIGH_Q, ELBOW_MIN, GRASP_DZ, GRASP_BACK, TABLE, PLATE)

UP = np.array([0.0, 0.0, 1.0])
DEBUG = False
W_NAT = 1.0
W_ROLL = 0.5
PULL = 0.0
G_PRE_UP = 0.06
G_PRE_BACK = 0.08
B_PRE_UP = 0.05
B_LIFT = 0.0
B_LIFT_ASIDE = 0.0
B_PRE_BACK = 0.07


def geom(blk, g0, block_z):
    d = np.r_[g0[:2] - blk[:2], 0.0]
    d /= np.linalg.norm(d)
    perp = np.array([-d[1], d[0], 0.0])
    z = block_z + GRASP_DZ
    return d, perp, z


_BC_CACHE = {}


def base_candidates(robot_base, target, d, table, seed_q=HIGH_Q):
    """IK-feasible base poses for reaching target with approach d, sorted by cost."""
    key = (tuple(np.round(np.asarray(target, float), 3)), tuple(np.round(np.asarray(d, float), 3)),
           tuple(np.round(table, 3)))
    sols = _BC_CACHE.get(key)
    if sols is None:
        sols = []
        for b0 in side_seeds(table, target):
            c, s_ = np.cos(b0[2]), np.sin(b0[2])
            shx = b0[0] - 0.05 * c - 0.188 * s_; shy = b0[1] - 0.05 * s_ + 0.188 * c
            if np.hypot(target[0] - shx, target[1] - shy) > 1.15:
                continue
            b, q, err = ik(target, d, b0, seed_q, free_base=True, base_nom=b0, elbow_min=ELBOW_MIN,
                           table=table, n_restarts=IK_RESTARTS_BC)
            if err > 5e-3 or rect_pen(b, 0.36, table) > 0:
                continue
            if any(np.linalg.norm(c2[:2] - b[:2]) < 0.08 and abs(wrap(c2[2] - b[2])) < 0.25 for c2, _ in sols):
                continue
            sols.append((b, q))
        _BC_CACHE[key] = sols
    cands = []
    for b, q in sols:
        sh = fk_points(b, q)[0]
        nat = target[:2] - sh[:2]
        ang = abs(wrap(np.arctan2(nat[1], nat[0]) - np.arctan2(d[1], d[0])))
        cost = route_len(robot_base, b, table) + W_NAT * ang + W_ROLL * abs(q[2])
        cands.append((cost, b, q))
    cands.sort(key=lambda c: c[0])
    return [(b, q) for _, b, q in cands]


IK_RESTARTS_BC = 6


def solve_chain(b, q, p, dp, seq, first_ik=True):
    """seq: list of (name, target, grip, dir). If first_ik, the first entry is reached by plain IK
    from configuration q (joint move); otherwise everything is Cartesian from (b,q,p,dp)."""
    segs = []
    if first_ik:
        name, tgt, g, dd = seq[0]
        _, q1, err = ik(tgt, dd, b, q, free_base=False, elbow_min=ELBOW_MIN)
        if err > 5e-3:
            b1, q1, err = ik(tgt, dd, b, q, free_base=True, base_nom=b, w_base=0.05,
                             elbow_min=ELBOW_MIN, table=TABLE)
            if err > 5e-3 or rect_pen(b1, 0.35, TABLE) > 0:
                if DEBUG: print('   first ik fail', name, err)
                return None
            b = b1
        q = q1
        if arm_hits(b, q) is not None:
            if DEBUG: print('   first hit', name, arm_hits(b, q))
            return None
        segs.append(dict(name=name, configs=[(b, q)], grip=g))
        p, dp = tgt, dd
        seq = seq[1:]
    for name, tgt, g, dd in seq:
        qs = cart_path(b, q, p, tgt, dp, d1=dd)
        if qs is None:
            if DEBUG: print('   chain fail at', name)
            return None
        segs.append(dict(name=name, configs=qs, grip=g))
        b, q = qs[-1]
        p, dd_ = tgt, dd
        dp = dd
    return segs


def aside_spots(cb, cg, d, perp, sh, table, plate):
    s = -np.sign((sh - cb) @ perp) or 1.0
    out = []
    for ss in (s, -s):
        found = None
        for dist in (0.2, 0.24, 0.16, 0.28):
            for back in (0.0, 0.05):
                a = cb + dist * ss * perp - back * d
                if valid_spot(a, table, plate, [cg]):
                    found = a
                    break
            if found is not None:
                break
        if found is not None:
            out.append(found)
    return out


def blocker_options(robot_base, blk, g0, block_z, table=TABLE, plate=PLATE, max_opts=12):
    """Yields (name, segs) for removing the blocker. segs[0] is a single pre_hi config; the
    executor prepends travel."""
    d, perp, z = geom(blk, g0, block_z)
    cb = np.array([blk[0], blk[1], z]); cg = np.array([g0[0], g0[1], z])
    mid = 0.5 * (cb + cg)
    n = 0
    for b, qg in base_candidates(robot_base, mid, d, table):
        sh = fk_points(b, qg)[0]
        asides = aside_spots(cb, cg, d, perp, sh, table, plate)
        if not asides:
            continue
        nat = cb[:2] - sh[:2]
        nat_ang = np.arctan2(nat[1], nat[0]); d_ang = np.arctan2(d[1], d[0])
        phis = [0.0, np.pi / 2, -np.pi / 2, 0.2, -0.2]
        phis.sort(key=lambda ph: abs(wrap(d_ang + ph - nat_ang)))
        found = False
        for phi in phis:
            db = _rot2(d, phi)
            p_pull = cb - PULL * d + 0.005 * UP
            seq = [("pre_hi", cb - B_PRE_BACK * db + B_PRE_UP * UP, 0.0, db),
                   ("pre", cb - B_PRE_BACK * db, 0.0, db),
                   ("grasp_b", cb, -1.0, db),
                   ("pull", p_pull, 0.0, db)]
            if DEBUG: print('  base', b.round(2), 'phi', round(phi, 2))
            pre = solve_chain(b, qg, None, None, seq)
            if pre is None:
                continue
            b1, q1 = pre[-1]['configs'][-1]
            for aside in asides:
                rest = solve_chain(b1, q1, p_pull, db, ([("blift", p_pull + B_LIFT * UP, 0.0, db)] if B_LIFT > 0 else []) + [("aside", aside + 0.003 * UP + B_LIFT_ASIDE * UP, 1.0, db),
                                                        ("retract", aside - RETRACT_BACK * db + RETRACT_UP * UP, 0.0, db)],
                                   first_ik=False)
                if DEBUG: print('   aside', aside.round(2), rest is not None)
                if rest is None:
                    continue
                n += 1
                found = True
                yield ('b%.2f_%.2f_%.2f_phi%.2f' % (b[0], b[1], b[2], phi), pre + rest)
                if n >= max_opts:
                    return
                break
            if found:
                break


_DROP_CACHE = {}


def _reachable_seeds(table, tgt, lim=1.0):
    out = []
    for b0 in side_seeds(table, tgt):
        c, s_ = np.cos(b0[2]), np.sin(b0[2])
        shx = b0[0] - 0.05 * c - 0.188 * s_; shy = b0[1] - 0.05 * s_ + 0.188 * c
        if np.hypot(tgt[0] - shx, tgt[1] - shy) <= lim:
            out.append(b0)
    return out


def _nat_dir(b, tgt, d):
    """Approach direction from base b's shoulder toward tgt, keeping d's pitch."""
    c, s_ = np.cos(b[2]), np.sin(b[2])
    shx = b[0] - 0.05 * c - 0.188 * s_; shy = b[1] - 0.05 * s_ + 0.188 * c
    n = np.array([tgt[0] - shx, tgt[1] - shy, 0.0])
    n /= np.linalg.norm(n) + 1e-12
    dh = np.linalg.norm(d[:2])
    return n * dh + np.array([0.0, 0.0, d[2]])


def drop_base(b, q, drops, d, table, grasp_back=None):
    """drops: block-centre targets (offset applied here per approach direction)."""
    key = (tuple(np.round(d, 2)), tuple(np.round(drops[0], 2)) if drops else None)
    if key in _DROP_CACHE:
        res = _DROP_CACHE[key]
    else:
        res = []
        gb = GRASP_BACK if grasp_back is None else grasp_back
        for dc in drops[:4]:
            for b0 in _reachable_seeds(table, dc):
                for dd in (_nat_dir(b0, dc, d), d):
                    dh = dd[:2] / np.linalg.norm(dd[:2])
                    dp = dc - gb * np.r_[dh, 0.0]
                    b2, q2, err = ik(dp, dd, b0, q, free_base=True, base_nom=b0, elbow_min=ELBOW_MIN,
                                     table=table, n_restarts=2)
                    if err > 5e-3 or rect_pen(b2, 0.36, table) > 0 or arm_hits(b2, q2) is not None:
                        continue
                    res.append((b2, q2))
                    break
            if res:
                break
        _DROP_CACHE[key] = res
    best = None
    for b2, q2 in res:
        L = route_len(b, b2, table)
        if best is None or L < best[0]:
            best = (L, b2, q2)
    return best


def green_chain_from(b, q, d, cg, z, table, plate, avoid, first_ik):
    """Chain: approach on axis, grasp, lift, then drop (same base or carry)."""
    lift = z + 0.16
    if first_ik:
        seq = [("g_pre", cg - G_PRE_BACK * d + G_PRE_UP * UP, 0.0, d),
               ("axis", cg - G_PRE_BACK * d, 0.0, d),
               ("grasp_g", cg - GRASP_BACK * d, -1.0, d),
               ("out", cg + (lift - z) * UP, 0.0, d)]
        segs = solve_chain(b, q, None, None, seq, first_ik=True)
    else:
        p = fk_world(b, q)[0] if False else None
        segs = None
    if segs is None:
        return None
    b, q = segs[-1]['configs'][-1]
    p = cg + (lift - z) * UP
    centres = drop_candidates(plate, avoid, lift, cg)
    for dc in centres[:4]:
        for dd in (d, _nat_dir(b, dc, d)):
            dh = dd[:2] / np.linalg.norm(dd[:2])
            dp = dc - GRASP_BACK * np.r_[dh, 0.0]
            qs = cart_path(b, q, p, dp, d, step=0.04, d1=dd)
            if qs is not None:
                segs.append(dict(name="drop", configs=qs, grip=1.0))
                return segs
    qs = cart_path(b, q, p, p + np.array([0, 0, 0.1]), d, step=0.04)
    if qs is not None:
        segs.append(dict(name="carry_lift", configs=qs, grip=0.0))
        b, q = qs[-1]
    best = drop_base(b, q, centres, d, table)
    if best is None:
        return None
    _, b2, q2 = best
    segs.append(dict(name="carry", configs=travel_configs(b, q, b2, q2, table, ignore=("green0",)),
                     grip=1.0))
    return segs


GREEN_VARIANTS = [(0.0, 0.0), (0.0, 0.3), (0.2, 0.0), (-0.2, 0.0), (0.2, 0.3), (-0.2, 0.3)]


def sim_steps(b, q, cfgs):
    """Number of steps _follow would take along cfgs from (b, q) (no collisions)."""
    b = np.asarray(b, float); q = np.asarray(q, float)
    n = 0; i = -1
    while i < len(cfgs) - 1:
        j = i
        while j + 1 < len(cfgs):
            bb, qq = cfgs[j + 1]
            db = np.r_[bb[:2] - b[:2], wrap(bb[2] - b[2])]
            if np.abs(jdist(q, qq)).max() <= 0.2 + 1e-9 and np.abs(db).max() <= 0.2 + 1e-9:
                j += 1
            else:
                break
        if j == i:
            j = i + 1
            bb, qq = cfgs[j]
            db = np.r_[bb[:2] - b[:2], wrap(bb[2] - b[2])]
            m = np.abs(np.r_[db, jdist(q, qq)]).max()
            k = int(np.ceil(m / 0.2 - 1e-9))
            n += max(k, 1)
        else:
            n += 1
        b, q = np.asarray(cfgs[j][0], float), np.asarray(cfgs[j][1], float)
        i = j
    return n


def est_segs_steps(cur_b, cur_q, segs):
    b0, q0 = segs[0]['configs'][-1]
    tc = travel_any(cur_b, cur_q, b0, q0)
    n = sim_steps(cur_b, cur_q, tc)
    if EST_HIT_PEN > 0:
        for bb, qq in tc:
            if arm_hits(bb, qq) is not None:
                n += EST_HIT_PEN
                break
    if not EST_MERGED:
        b, q = b0, q0
        for w in segs[1:]:
            n += sim_steps(b, q, w['configs'])
            b, q = w['configs'][-1]
        return n
    # merged execution: travel + segments joined until a grip change / carry
    chunks = []; cur = list(tc)
    for w in segs[1:]:
        if w['name'] in ('carry', 's_carry', 'carry_lift'):
            if cur:
                chunks.append(cur)
            chunks.append(list(w['configs'])); cur = []
            continue
        cur += list(w['configs'])
        if w['grip'] != 0:
            chunks.append(cur); cur = []
    if cur:
        chunks.append(cur)
    n = n - sim_steps(cur_b, cur_q, tc)
    b, q = np.asarray(cur_b, float), np.asarray(cur_q, float)
    for c in chunks:
        n += sim_steps(b, q, c)
        b, q = c[-1]
    return n


GREEN_EXTRA_BASES = 8
EST_HIT_PEN = 0
EST_MERGED = True
GREEN_EXTRA_NOCARRY = 3
RETRACT_UP = 0.18
RETRACT_BACK = 0.05
GREEN_SEED_CUR = False


def green_options(cur_base, g0, blk_pos, d, block_z, table=TABLE, plate=PLATE, max_opts=8,
                  cur_q=None, deadline=None):
    """Yields (name, segs) for grasping green0 roughly along d and dropping on plate.
    Options needing a carry are held back while a few more bases are tried; the best by
    estimated step count is yielded first."""
    import time as _time
    z = block_z + GRASP_DZ
    cg = np.array([g0[0], g0[1], z])
    tgt = cg - 0.08 * d
    n = 0
    pending = []
    extra = None

    def gen_cands():
        seed = HIGH_Q if (cur_q is None or not GREEN_SEED_CUR) else np.asarray(cur_q, float)
        _, qc, err = ik(tgt, d, cur_base, seed, free_base=False, elbow_min=ELBOW_MIN)
        if err < 5e-3:
            yield (np.asarray(cur_base, float), qc)
        for c in base_candidates(cur_base, tgt, d, table):
            yield c

    def flush():
        pending.sort(key=lambda t: t[0])
        out = [(nm, sg) for _, nm, sg in pending]
        pending.clear()
        return out
    for b, q in gen_cands():
        for yo, pt in GREEN_VARIANTS:
            dv = _rot2(d, yo)
            d3 = np.cos(pt) * dv - np.sin(pt) * UP
            segs = green_chain_from(b, q, d3, cg, z, table, plate, [blk_pos], True)
            if segs is None:
                continue
            nm = 'g%.2f_%.2f_%.2f_y%.1f_p%.1f' % (b[0], b[1], b[2], yo, pt)
            carry = any(w['name'] == 'carry' for w in segs)
            if cur_q is not None:
                est = est_segs_steps(np.asarray(cur_base, float), np.asarray(cur_q, float), segs)
                pending.append((est, nm, segs))
                if DEBUG: print("GOPT", nm, est, carry)
                if extra is None:
                    extra = GREEN_EXTRA_BASES if carry else GREEN_EXTRA_NOCARRY
            else:
                pending.append((0, nm, segs))
                extra = 0
            break
        if extra is not None:
            if extra <= 0 or (deadline is not None and _time.time() > deadline):
                for o in flush():
                    n += 1
                    yield o
                extra = None if n < max_opts else extra
            else:
                extra -= 1
        if n >= max_opts:
            return
    for o in flush():
        yield o


FAR = (-4.5, 0.0, 0.3, 0.6)


def table_of(b):
    return FAR if b[0] < 0 else TABLE


def travel_any(b_from, q_from, b_to, q_to):
    """Travel possibly between the two tables via hub points on their inner sides."""
    ta, tb = table_of(b_from), table_of(b_to)
    if ta == tb:
        return travel_configs(b_from, q_from, b_to, q_to, ta)
    sgn = 1.0 if ta[0] > 0 else -1.0
    yaw = np.pi if sgn > 0 else 0.0
    hub_a = np.array([sgn * 3.75, 0.0, yaw])
    hub_b = np.array([-sgn * 3.75, 0.0, yaw])
    c1 = travel_configs(b_from, q_from, hub_a, q_from, ta, step=0.2)
    c2 = [(hub_a + (hub_b - hub_a) * t, q_from) for t in np.linspace(0, 1, 40)[1:]]
    c3 = travel_configs(hub_b, q_from, b_to, q_to, tb)
    return c1 + c2 + c3


def spare_options(cur_base, spares, block_z, plate=PLATE, max_opts=4):
    """spares: list of (x, y, yaw). Yields (name, segs) where segs[0] is a pre_hi config near
    the far table; the final segment is a carry to the near table + drop (grip open)."""
    n = 0
    order = sorted(spares, key=lambda s: -s[0])
    z = block_z + GRASP_DZ
    for sx, sy, syaw in order:
        cs = np.array([sx, sy, z])
        dirs = [syaw + k * np.pi / 2 for k in range(4)]
        dirs.sort(key=lambda a: abs(wrap(a - np.pi)))
        done = False
        for a in dirs[:3]:
            dd = np.array([np.cos(a), np.sin(a), 0.0])
            for b, qg in base_candidates(cur_base, cs, dd, FAR)[:2]:
                lift = z + 0.16
                seq = [("s_prehi", cs - 0.10 * dd + 0.10 * UP, 0.0, dd),
                       ("s_pre", cs - 0.10 * dd, 0.0, dd),
                       ("s_grasp", cs, -1.0, dd),
                       ("s_lift", cs + 0.16 * UP, 0.0, dd)]
                segs = solve_chain(b, qg, None, None, seq)
                if segs is None:
                    continue
                b1, q1 = segs[-1]['configs'][-1]
                best = None
                for dp in drop_candidates(plate, [], lift, np.array([3.8, 0.0, 0.0]))[:6]:
                    for b0 in side_seeds(TABLE, dp):
                        sh = b0[:2]
                        v = dp[:2] - sh
                        d2 = np.r_[v / np.linalg.norm(v), 0.0]
                        b2, q2, err = ik(dp, d2, b0, q1, free_base=True, base_nom=b0,
                                         elbow_min=ELBOW_MIN, table=TABLE)
                        if err > 5e-3 or rect_pen(b2, 0.36, TABLE) > 0:
                            continue
                        L = route_len(np.array([3.75, 0.0, np.pi]), b2, TABLE)
                        if best is None or L < best[0]:
                            best = (L, b2, q2)
                    if best is not None:
                        break
                if best is None:
                    continue
                _, b2, q2 = best
                segs.append(dict(name="s_carry", configs=travel_any(b1, q1, b2, q2), grip=1.0))
                n += 1
                yield ('spare%.2f_%.2f_a%.2f' % (sx, sy, a), segs)
                done = True
                break
            if done or n >= max_opts:
                break
        if n >= max_opts:
            return


BLK_EXTRA = 5


def blocker_options_sorted(robot_base, cur_q, blk, g0, block_z, deadline=None, **kw):
    """Collect the first BLK_EXTRA+1 blocker options, yield them sorted by estimated steps,
    then the rest in generator order."""
    import time as _time
    gen = blocker_options(robot_base, blk, g0, block_z, **kw)
    first = []
    for nm, segs in gen:
        est = est_segs_steps(np.asarray(robot_base, float), np.asarray(cur_q, float), segs)
        if DEBUG: print("BOPT", nm, est)
        first.append((est, nm, segs))
        if len(first) > BLK_EXTRA or (deadline is not None and _time.time() > deadline):
            break
    first.sort(key=lambda t: t[0])
    for _, nm, segs in first:
        yield nm, segs
    for o in gen:
        yield o
