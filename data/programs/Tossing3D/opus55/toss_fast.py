"""TOSS primitive for Tossing3D (tidybot) -- faster variant of toss.py (same throw & aiming).

Changes vs toss.py (step savings), everything from the lift onwards is identical to toss.py:
  * pick: the base drives to the pick pose while the arm already servos to the pre-grasp pose
    (IK planned for the goal base pose); no 4-step "still" wait; pre-grasp tolerance 0.01.
  * descent: 3 waypoints (dz 0.043, 0.021, 0) instead of 8 (the final approach matches toss.py,
    so the grasp depth -- which the throw calibration is very sensitive to -- is unchanged).
  * gripper close 5 steps instead of 8 (fingers settle in 2).
  * after the release of a non-final cube, the next pick starts immediately (no landing wait).
  * cube order minimises base driving not hidden under the wind-up (start->1st cube, throw->2nd).
  NOT changed (tested: they alter cube slip in the gripper during the wind-up -> landing error):
  lift (0.30 m), wind-up speed (0.08 rad/step), settle, swing, aiming.
Per-cube phases (see ph() tags): p_base, p_above, p_desc, p_close, p_lift, t_wind, t_settle,
t_swing, t_wait. Tunables in dict P (env-var overridable, for experiments only).

Usage (generator-style policy):
    g = toss_all(obs)
    a = next(g)
    while True:
        obs, r, term, trunc, info = env.step(a)
        if term or trunc: break
        a = g.send(obs)
"""
import numpy as np
import kin

J = ['pos_arm_joint%d' % i for i in range(1, 8)]
DT = 0.1
OPEN, CLOSE = 0.0, 1.0
PHASE = ['']
THROWS = []

def ph(n):
    PHASE[0] = n


def rq(o):
    R = o.get_object_from_name('robot')
    return np.array([o.get(R, j) for j in J])


def rv(o):
    R = o.get_object_from_name('robot')
    return np.array([o.get(R, j.replace('pos', 'vel')) for j in J])


def rb(o):
    R = o.get_object_from_name('robot')
    return np.array([o.get(R, 'pos_base_x'), o.get(R, 'pos_base_y'), o.get(R, 'pos_base_rot')])


def objpos(o, name):
    C = o.get_object_from_name(name)
    return np.array([o.get(C, f) for f in ['x', 'y', 'z']])


def objvel(o, name):
    C = o.get_object_from_name(name)
    return np.array([o.get(C, f) for f in ['vx', 'vy', 'vz']])


def obj_yaw(o, name):
    C = o.get_object_from_name(name)
    qw, qx, qy, qz = [o.get(C, f) for f in ['qw', 'qx', 'qy', 'qz']]
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def down_R(yaw):
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0, 0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])


def arm_act(obs, qt, grip, vmax=0.35, base=(0, 0, 0)):
    a = kin.servo_action(rq(obs), rv(obs), qt, vmax=vmax, grip=grip)
    a[:3] = base
    return a


def base_delta(obs, goal):
    b = rb(obs)
    d = np.array([goal[0] - b[0], goal[1] - b[1], wrap(goal[2] - b[2])])
    return np.clip(d, -0.1, 0.1)


# ------------------------------------------------------------------ pick
_NOENV = {}  # tuning params frozen at their defaults
P = dict(
    REACH=0.55,
    N_CLOSE=int(_NOENV.get('N_CLOSE', 5)),
    N_DESC=int(_NOENV.get('N_DESC', 4)),       # streamed descent waypoints (1 step each)
    LIFT_DZ=float(_NOENV.get('LIFT_DZ', 0.30)),
    LIFT_TOL=float(_NOENV.get('LIFT_TOL', 0.005)),
    WIND_VMAX=float(_NOENV.get('WIND_VMAX', 0.08)),
    WIND_ACC=float(_NOENV.get('WIND_ACC', 1.0)),
    DESC_END=float(_NOENV.get('DESC_END', 0.03)),
    DESC_VMAX=float(_NOENV.get('DESC_VMAX', 0.05)),
    DESC_TOL=float(_NOENV.get('DESC_TOL', 0.003)),
    DESC_VTOL=float(_NOENV.get('DESC_VTOL', 0.02)),
    STILL=int(_NOENV.get('STILL', 0)),
    STILL_HOLD=int(_NOENV.get('STILL_HOLD', 1)),
    DESC_ORIG=int(_NOENV.get('DESC_ORIG', 1)),
    DESC_SKIP=int(_NOENV.get('DESC_SKIP', 5)),
    DESC_UTOL=float(_NOENV.get('DESC_UTOL', 0.005)),
    WIND_VMAX_Q4=float(_NOENV.get('WIND_VMAX_Q4', 0.08)),
    ORDER_OPT=int(_NOENV.get('ORDER_OPT', 1)),
    POST_SWING=int(_NOENV.get('POST_SWING', 0)),
)


def _grasp_plan(bx, by, br, q0, c0, cyaw):
    best = None
    for k in range(4):
        yk = wrap(cyaw + k * np.pi / 2)
        qk, ek = kin.ik_multi(bx, by, br, q0, c0 + [0, 0, 0.15], R_des=down_R(yk))
        cost = ek + 0.01 * abs(wrap(yk - br))
        if best is None or cost < best[0] - 1e-4:
            best = (cost, yk, qk)
    return best[1], best[2]


def pick(obs, name):
    """Generator: grasp object `name` and lift it. Base driving overlaps the arm motion to the
    pre-grasp pose (planned for the goal base pose, re-planned for the actual pose on arrival)."""
    R = P['REACH']
    ph('p_base')
    c0 = objpos(obs, name)
    cyaw = obj_yaw(obs, name)
    b = rb(obs)
    d = c0[:2] - b[:2]
    yaw_goal = np.arctan2(d[1], d[0])
    goal = np.array([c0[0] - R * np.cos(yaw_goal), c0[1] - R * np.sin(yaw_goal), yaw_goal])
    far = np.linalg.norm(d) > 0.3
    q_start = rq(obs).copy()
    _, q_pre = _grasp_plan(*goal, q_start, c0, cyaw)
    for _ in range(80):
        b = rb(obs)
        base_ok = np.linalg.norm(goal[:2] - b[:2]) < 0.02 and abs(wrap(goal[2] - b[2])) < 0.03
        arm_ok = np.abs(rq(obs) - q_pre).max() < 0.05
        if base_ok and arm_ok:
            break
        obs = yield arm_act(obs, q_pre, OPEN, base=base_delta(obs, goal))
    ph('p_still')
    for _ in range(P['STILL']):
        obs = yield arm_act(obs, rq(obs) if P['STILL_HOLD'] else q_pre, OPEN)
    ph('p_above')
    bx, by, br = rb(obs)
    c0 = objpos(obs, name)
    cyaw = obj_yaw(obs, name)
    yaw, q_above = _grasp_plan(bx, by, br, q_start, c0, cyaw)

    def goto(qt, grip, tol=0.005, maxn=60, vmax=0.35):
        nonlocal obs
        for _ in range(maxn):
            obs = yield arm_act(obs, qt, grip, vmax=vmax)
            if np.abs(rq(obs) - qt).max() < tol:
                break

    yield from goto(q_above, OPEN, tol=0.01)
    ph('p_desc')
    qd = q_above
    if P['DESC_ORIG']:
        for dz in np.linspace(0.15, 0.0, 8)[P['DESC_SKIP']:]:
            qd, _ = kin.ik(bx, by, br, rq(obs), c0 + np.array([0, 0, dz]), R_des=down_R(yaw))
            yield from goto(qd, OPEN, maxn=20, tol=0.005 if dz < 0.05 else P['DESC_UTOL'])
    for dz in [] if P['DESC_ORIG'] else np.linspace(0.15, P['DESC_END'], P['N_DESC'] + 1)[1:]:
        bx, by, br = rb(obs)
        qd, _ = kin.ik(bx, by, br, qd, c0 + np.array([0, 0, dz]), R_des=down_R(yaw))
        obs = yield arm_act(obs, qd, OPEN)
    bx, by, br = rb(obs)
    qd, _ = kin.ik(bx, by, br, qd, c0, R_des=down_R(yaw))
    for _ in range(0 if P['DESC_ORIG'] else 20):
        obs = yield arm_act(obs, qd, OPEN, vmax=P['DESC_VMAX'])
        if np.abs(rq(obs) - qd).max() < P['DESC_TOL'] and np.abs(rv(obs)).max() < P['DESC_VTOL']:
            break
    ph('p_close')
    for _ in range(P['N_CLOSE']):
        obs = yield arm_act(obs, qd, CLOSE)
    ph('p_lift')
    bx, by, br = rb(obs)
    qu, _ = kin.ik(bx, by, br, rq(obs), c0 + np.array([0, 0, P['LIFT_DZ']]), R_des=down_R(yaw))
    yield from goto(qu, CLOSE, tol=P['LIFT_TOL'])
    return obs


# ------------------------------------------------------------------ helpers for throwing
def wind_act(obs, qt, grip, vmax, acc, base=(0, 0, 0)):
    """Trapezoidal (accel-limited) joint servo toward qt; acc in rad/step^2."""
    q, v = rq(obs), rv(obs) * DT
    err = qt - q
    lim = np.minimum(vmax, np.sqrt(2 * acc * np.abs(err)))
    lim = np.minimum(lim, np.abs(v) + acc)
    y = np.clip(0.8 * err, -lim, lim)
    a = np.zeros(18, np.float32)
    a[11:18] = (y + 0.59 * rv(obs) * DT) / 0.059
    a[10] = grip
    a[:3] = base
    return a


def plane_q(obs, q2, q4, q6, q7):
    """Joint target in the vertical throwing plane (q1=0, q3=pi, q5=0) with continuous
    joints unwrapped relative to the current configuration."""
    q = rq(obs)
    tgt = np.array([0.0, q2, np.pi, q4, 0.0, q6, q7])
    for i in (0, 2, 4, 6):
        tgt[i] = q[i] + wrap(tgt[i] - q[i])
    return tgt


def vel_act(obs, dq_des, hold_q, grip, base=(0, 0, 0), mask=(1, 3, 5)):
    """Deadbeat velocity command: joints in `mask` move dq_des[i] rad this step,
    others are servoed to hold_q."""
    a = kin.servo_action(rq(obs), rv(obs), hold_q, vmax=0.35, grip=grip)
    v = rv(obs)
    for i in mask:
        a[11 + i] = (dq_des[i] + 0.59 * v[i] * DT) / 0.059
    a[:3] = base
    return a


# ------------------------------------------------------------------ throw
# Wind-up pose (q2, q4, q6) in the throwing plane (q1=0, q3=pi, q5=0, q7=0): arm cocked back-up.
Q_WIND = (-0.5, 0.8, 0.8)
Q7_WIND = 0.0
SWING_DIR = np.array([0, 1, 0, -1, 0, -1, 0], float)   # q2 up, q4/q6 down = forward overhand
RELEASE_STEP = 1      # gripper opens on the 2nd swing step (cube leaves early in that step)
N_SWING = 3           # velocity-controlled swing steps, then hold
# Calibration (base yaw 0, cube thrown in +x): swing speed w [rad/s per joint] -> horizontal
# distance D from base x to the cube's floor landing point. 30 seeds per node; D also depends
# on the cube's lateral offset in the gripper (tool-frame y, metres): D += CAL_OFFY_SLOPE*offy.
# Residual std ~1.5-2.5 cm. Only w in [3.2,3.35] give a flight time whose landing phase w.r.t. the
# 0.1 s control step lets the env register the cube inside the bin before it punches through
# the (thin) bin floor; lower w failed ~10-20% of the time.
CAL_W = np.array([3.2, 3.25, 3.3])
CAL_D = np.array([2.6651, 2.7771, 2.9228])
CAL_OFFY_SLOPE = 6.33          # m of D per m of tool-y offset
LAT_OFF = 0.0         # lateral landing offset (world y) relative to base y (measured ~0 +-3mm)
AIM_BEYOND = 0.06     # aim past the bin center: 100% success window measured for aim in [0.01, 0.13]
BX_MIN, BX_MAX = -0.05, 0.75   # allowed base x during the throw (barrier blocks base at ~1.0)
PREF_W = (3.25, 3.2, 3.3)      # node preference order (3.25 has the smallest spread)


def plan_throw(bin_xy, offy=0.0, aim=None):
    """Return (base_goal(x,y,yaw), w) for landing at bin center + aim along x."""
    aim = AIM_BEYOND if aim is None else aim
    L = bin_xy[0] + aim
    for w in PREF_W:
        D = np.interp(w, CAL_W, CAL_D) + CAL_OFFY_SLOPE * offy
        bx = L - D
        if BX_MIN <= bx <= BX_MAX:
            return np.array([bx, bin_xy[1] - LAT_OFF, 0.0]), w
    # fallback: nearest feasible
    w = 3.2 if L - CAL_D[0] < BX_MIN else 3.3
    D = np.interp(w, CAL_W, CAL_D) + CAL_OFFY_SLOPE * offy
    return np.array([np.clip(L - D, BX_MIN, 0.9), bin_xy[1] - LAT_OFF, 0.0]), w


def throw(obs, name, bin_name='bin_0', w=None, base_goal=None, wind_vmax=None, log=None, aim=None, last=True):
    """Generator: (object already grasped) drive base to throw pose, wind up, swing, release,
    then wait until the object comes to rest. Returns final obs."""
    wind_vmax = P['WIND_VMAX'] if wind_vmax is None else wind_vmax
    wind_vmax = np.full(7, wind_vmax)
    wind_vmax[3] = max(wind_vmax[3], P['WIND_VMAX_Q4'])
    auto = base_goal is None or w is None
    w_fixed, bg_fixed = w, base_goal

    def replan():
        if not auto:
            return bg_fixed, w_fixed
        bg, ww = plan_throw(objpos(obs, bin_name)[:2], grasp_offset(obs, name)[1], aim=aim)
        return (bg if bg_fixed is None else bg_fixed), (ww if w_fixed is None else w_fixed)
    base_goal, w = replan()
    Qs = plane_q(obs, *Q_WIND, Q7_WIND)
    ph('t_wind')
    for i in range(200):
        base_goal, w = replan()
        obs = yield wind_act(obs, Qs, CLOSE, wind_vmax, P['WIND_ACC'], base=base_delta(obs, base_goal))
        if np.abs(rq(obs) - Qs).max() < 0.01 and np.linalg.norm((rb(obs) - base_goal)[:2]) < 0.005 \
                and abs(wrap(rb(obs)[2] - base_goal[2])) < 0.005:
            break
    ph('t_settle')
    for _ in range(30):   # settle precisely (release direction is sensitive to the start pose)
        base_goal, w = replan()
        obs = yield arm_act(obs, Qs, CLOSE, vmax=0.05, base=base_delta(obs, base_goal))
        if np.abs(rq(obs) - Qs).max() < 0.0025 and np.abs(rv(obs)).max() < 0.01:
            break
    if log is not None:
        log.append(np.r_[rb(obs) - base_goal, rq(obs) - Qs, grasp_offset(obs, name)])
    THROWS.append((name, grasp_offset(obs, name).round(4), w, np.abs(rq(obs) - Qs).max().round(4), np.abs(rv(obs)).max().round(3)))
    hold = rq(obs).copy()
    dq = SWING_DIR * w * DT
    ph('t_swing')
    for t in range(N_SWING):
        grip = OPEN if t >= RELEASE_STEP else CLOSE
        obs = yield vel_act(obs, dq, hold, grip)
        if log is not None:
            log.append(np.r_[objpos(obs, name), objvel(obs, name)])
    hold = rq(obs).copy()
    ph('t_wait')
    for t in range(40 if last else P['POST_SWING']):
        obs = yield arm_act(obs, hold, OPEN)
        p, v = objpos(obs, name), objvel(obs, name)
        if log is not None:
            log.append(np.r_[p, v])
        if p[2] < 0.3 and np.linalg.norm(v) < 0.05:
            break
    return obs


def grasp_offset(obs, name):
    """Object position relative to the FK tool point, in the tool frame."""
    p, R = kin.fk(*rb(obs), rq(obs))
    return R.T @ (objpos(obs, name) - p)


def cube_names(obs):
    return sorted([n for n in obs.get_object_names() if n.startswith('cube_')])


def in_bin(obs, name, bin_name='bin_0'):
    p, b = objpos(obs, name), objpos(obs, bin_name)
    return abs(p[0] - b[0]) < 0.13 and abs(p[1] - b[1]) < 0.13 and p[2] < 0.2


def find_bin(obs):
    names = sorted(n for n in obs.get_object_names() if n.startswith('bin'))
    return names[0] if names else 'bin_0'


def _drive_steps(frm, c):
    """Approx. base steps to reach the pick pose for a cube at c from base pose frm."""
    d = c[:2] - frm[:2]
    yaw = np.arctan2(d[1], d[0])
    g = np.array([c[0] - P['REACH'] * np.cos(yaw), c[1] - P['REACH'] * np.sin(yaw), yaw])
    return np.max(np.abs([g[0] - frm[0], g[1] - frm[1], wrap(g[2] - frm[2])])) / 0.1


def order_cubes(obs, todo, bin_name):
    """Order cubes to minimise base driving not hidden under the wind-up (pick-pose legs)."""
    b = rb(obs)
    if len(todo) <= 1 or not P['ORDER_OPT']:
        return sorted(todo, key=lambda n: np.linalg.norm(objpos(obs, n)[:2] - b[:2]))
    import itertools
    T, _ = plan_throw(objpos(obs, bin_name)[:2])
    best = None
    for perm in itertools.permutations(todo) if len(todo) <= 5 else [todo]:
        cost, frm = 0.0, b
        for n in perm:
            cost += _drive_steps(frm, objpos(obs, n))
            frm = T
        if best is None or cost < best[0]:
            best = (cost, list(perm))
    return best[1]


def toss_all(obs, bin_name=None):
    """Generator policy: pick and throw every cube not yet in the bin."""
    bin_name = bin_name or find_bin(obs)
    THROWS.clear()
    for _ in range(3):   # up to 3 attempts per cube
        todo = [n for n in cube_names(obs) if not in_bin(obs, n, bin_name)]
        if not todo:
            break
        todo = order_cubes(obs, todo, bin_name)
        for k, n in enumerate(todo):
            obs = yield from pick(obs, n)
            obs = yield from throw(obs, n, bin_name, last=(k == len(todo) - 1))
    ph('end_wait')
    while True:
        obs = yield arm_act(obs, rq(obs), OPEN)
