"""Probe box support/termination after a deterministic grasp (box-only seed 0)."""
import math
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from env_client import make_env
from probe_grasp_structured import command, val


LO = np.array([0., -.35, -math.pi, -2.5, 0., -.87, math.pi/2])
HI = LO + np.array([5.2, 2.41, 2.058, 2.66, 5.2, 2.23, 6.77])
RNG = np.random.default_rng(0)
ROUTE = []
for _ in range(28):
    q = RNG.uniform(LO, HI); r = RNG.uniform(.05, .95); ang = RNG.uniform(-math.pi, math.pi)
    ROUTE.append((q, r, ang, RNG.uniform(-math.pi, math.pi)))


def pose(s):
    return (np.array([val(s, 'box0', 'pose_' + c) for c in 'xyz']),
            np.array([val(s, 'box0', 'pose_q' + c) for c in 'xyzw']))


def jacobian(env, s, grip=-1):
    p0, q0 = pose(s); jmat = np.zeros((6, 7)); eps = .025
    for j in range(7):
        a = np.zeros(11); a[3+j] = eps; a[10] = grip
        sp, _, _, _, _ = env.step(a.astype(np.float32)); pp, qp = pose(sp)
        jmat[:3, j] = (pp-p0)/eps
        jmat[3:, j] = (Rotation.from_quat(qp)*Rotation.from_quat(q0).inv()).as_rotvec()/eps
        a[3+j] = -eps
        s, _, _, _, _ = env.step(a.astype(np.float32))
    return s, jmat


env = make_env(); s, _ = env.reset(seed=0, options={'object_count': 0})
goal_x = float(sys.argv[1]) if len(sys.argv) > 1 else .6
goal_y = float(sys.argv[2]) if len(sys.argv) > 2 else 0.
grasp_delta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.
do_regrasp = len(sys.argv) > 4 and sys.argv[4] == 'regrasp'
table_adjust = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != 'regrasp' else 0.
tx, ty = val(s, 'box0', 'pose_x'), val(s, 'box0', 'pose_y')
for idx in (24, 26, 27):
    q, r, ang, rot = ROUTE[idx]
    q = q.copy()
    if idx == 27:
        q[0] += grasp_delta
    s = command(env, s, [tx+r*math.cos(ang), ty+r*math.sin(ang), rot], q, 1., 35)
    s = command(env, s, [tx+r*math.cos(ang), ty+r*math.sin(ang), rot], q, -1., 2)
print('grasp', grasp_delta, val(s, 'robot', 'grasp_active'), [val(s, 'robot', 'grasp_tf_'+c) for c in 'xyz'])

# Lift while keeping the large box level.
for _ in range(18):
    s, jmat = jacobian(env, s)
    p, q = pose(s); twist = np.r_[0, 0, .06, -2*Rotation.from_quat(q).as_rotvec()]
    dq = jmat.T @ np.linalg.solve(jmat@jmat.T + .01*np.eye(6), twist)
    a = np.zeros(11); a[3:10] = np.clip(dq, -.08, .08); a[10] = -1
    s, _, _, _, _ = env.step(a.astype(np.float32))

# Carry by base motion; it translates the attached object exactly.
for _ in range(20):
    p, _ = pose(s); a = np.zeros(11); a[:2] = np.clip([goal_x-p[0], goal_y-p[1]], -.2, .2); a[10] = -1
    s, _, _, _, _ = env.step(a.astype(np.float32))

# Remove the few degrees of residual tilt from the lift before approaching the
# support plane.  Tilt changes the first-contact center height substantially.
for _ in range(8):
    p, q = pose(s)
    rv = Rotation.from_quat(q).as_rotvec()
    if np.linalg.norm(rv) < .001:
        break
    s, jmat = jacobian(env, s)
    dq = jmat.T @ np.linalg.solve(jmat@jmat.T + .003*np.eye(6), np.r_[0, 0, 0, -rv])
    a = np.zeros(11); a[3:10] = np.clip(dq, -.04, .04); a[10] = -1
    s, _, _, _, _ = env.step(a.astype(np.float32))
print('leveled', goal_x, goal_y, np.round(pose(s)[0], 4), np.round(pose(s)[1], 4))

if do_regrasp:
    # Approximate a world-up end-effector increment from the attached-object
    # Jacobian, detach at safe height, and close again from progressively higher
    # contacts on the stationary box.
    s, jmat = jacobian(env, s)
    up_dq = np.clip(jmat.T @ np.linalg.solve(jmat@jmat.T + .005*np.eye(6),
                                             np.r_[0, 0, .035, 0, 0, 0]), -.08, .08)
    a = np.zeros(11); a[10] = 1
    s, _, _, _, _ = env.step(a.astype(np.float32))
    for k in range(8):
        a = np.zeros(11); a[3:10] = up_dq; a[10] = 1
        s, _, _, _, _ = env.step(a.astype(np.float32))
        a[:] = 0; a[10] = -1
        s, _, _, _, _ = env.step(a.astype(np.float32))
        held = val(s, 'robot', 'grasp_active') > .5
        print('regrasp', k, held, [round(val(s, 'robot', 'grasp_tf_'+c), 4) for c in 'xyz'])
        if held:
            break
        a[10] = 1
        s, _, _, _, _ = env.step(a.astype(np.float32))

# Descend until contact blocks further motion.
last_dq = np.zeros(7)
for k in range(16):
    s, jmat = jacobian(env, s)
    p, q = pose(s); twist = np.r_[0, 0, -.05, -2*Rotation.from_quat(q).as_rotvec()]
    last_dq = np.clip(jmat.T @ np.linalg.solve(jmat@jmat.T + .01*np.eye(6), twist), -.08, .08)
    a = np.zeros(11); a[:2] = np.clip([goal_x-p[0], goal_y-p[1]], -.2, .2); a[3:10] = last_dq; a[10] = -1
    s, r, term, trunc, info = env.step(a.astype(np.float32))
    print('down', k, np.round(pose(s)[0], 4), 'held', val(s, 'robot', 'grasp_active'), 'term', term)
    if pose(s)[0][2] <= .509:
        break

# Detach, then continue the downward arm command to test whether open fingers push
# the unsupported box onto the top surface.
a = np.zeros(11); a[10] = 1
for k in range(3):
    s, r, term, trunc, info = env.step(a.astype(np.float32))
    print('open', k, np.round(pose(s)[0], 4), val(s, 'robot', 'grasp_active'), term)
if table_adjust:
    a = np.zeros(11); a[3] = table_adjust; a[10] = 1
    s, _, _, _, _ = env.step(a.astype(np.float32))
    a[:] = 0; a[10] = -1
    s, _, _, _, _ = env.step(a.astype(np.float32))
    print('table_regrasp', table_adjust, val(s, 'robot', 'grasp_active'),
          [round(val(s, 'robot', 'grasp_tf_'+c), 4) for c in 'xyz'])
for k in range(12):
    a = np.zeros(11); a[3:10] = last_dq; a[10] = -1 if table_adjust else 1
    s, r, term, trunc, info = env.step(a.astype(np.float32))
    print('push', k, np.round(pose(s)[0], 4), val(s, 'robot', 'grasp_active'), term)
env.close()
