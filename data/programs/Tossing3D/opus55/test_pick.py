"""Pick test: IK above cube (gripper pointing down), descend, close, lift.
Also logs (base, q, cube pos) for FK calibration into pick_log.npz."""
import sys
import numpy as np
from env_client import make_env
import kin

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
J = ['pos_arm_joint%d' % i for i in range(1, 8)]
KV = 0.3645      # steady joint vel (rad/s) per unit of vel-target action
DT = 0.1

env = make_env()
obs, info = env.reset(seed=SEED)
R = obs.get_object_from_name('robot')
C = obs.get_object_from_name('cube_0')
log = []


def rq(o):
    return np.array([o.get(R, j) for j in J])


def rv(o):
    return np.array([o.get(R, j.replace('pos', 'vel')) for j in J])


def rb(o):
    return o.get(R, 'pos_base_x'), o.get(R, 'pos_base_y'), o.get(R, 'pos_base_rot')


def cube(o):
    return np.array([o.get(C, f) for f in ['x', 'y', 'z']])


def cube_quat(o):
    return np.array([o.get(C, f) for f in ['qw', 'qx', 'qy', 'qz']])


def step(qt, grip, n=1, vmax=0.35):
    """Servo arm joints toward qt using vel targets; returns obs."""
    global obs
    for _ in range(n):
        q = rq(obs)
        err = qt - q
        a = np.zeros(18, np.float32)
        # identified per-step model: dq_k = 0.059*u_k - 0.59*dq_{k-1} (u = vel target)
        y = np.clip(0.8 * err, -vmax, vmax)
        a[11:18] = (y + 0.59 * rv(obs) * DT) / 0.059
        a[10] = grip
        obs, *_ = env.step(a)
        log.append(np.concatenate([rb(obs), rq(obs), cube(obs), [grip]]))
    return obs


def goto(qt, grip, tol=0.005, maxn=60, vmax=0.35):
    for i in range(maxn):
        step(qt, grip, vmax=vmax)
        if np.abs(rq(obs) - qt).max() < tol:
            break
    return i


def down_R(yaw):
    # tool z = -world z, tool x = (cos yaw, sin yaw, 0)
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0, 0, -1.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


BASEMOVE = len(sys.argv) > 3
if BASEMOVE:  # move/rotate base first to validate FK in a non-trivial base pose
    for _ in range(10):
        a = np.zeros(18, np.float32); a[:3] = [0.01, -0.02, 0.04]
        obs, *_ = env.step(a)
    for _ in range(10):
        obs, *_ = env.step(np.zeros(18, np.float32))
    print('base', np.round(rb(obs), 3))
bx, by, br = rb(obs)
c0 = cube(obs)
qw, qx, qy, qz = cube_quat(obs)
cyaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
# pick yaw close to current tool yaw mod pi/2
best = None
for k in range(4):   # cube is 90deg-symmetric; choose the yaw with the best IK
    yk = cyaw + k * np.pi / 2
    qk, ek = kin.ik_multi(bx, by, br, rq(obs), c0 + np.array([0, 0, 0.15]), R_des=down_R(yk))
    if best is None or ek < best[0] - 1e-4:
        best = (ek, yk)
yaw = best[1]
print('cube', c0.round(3), 'yaw', round(cyaw, 3))

q = rq(obs)
OPEN, CLOSE = 0.0, 1.0
p_above = c0 + np.array([0, 0, 0.15])
q_above, e = kin.ik_multi(bx, by, br, q, p_above, R_des=down_R(yaw))
print('ik above err', e, q_above.round(3))
n = goto(q_above, OPEN)
p, _ = kin.fk(*rb(obs), rq(obs))
print('reached above in', n, 'fk', p.round(3), 'cube', cube(obs).round(3))

# descend in small increments, watch cube
zoff = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0  # extra height offset of grasp point above cube center
for dz in np.linspace(0.15, zoff, 8):
    qd, e = kin.ik(bx, by, br, rq(obs), c0 + np.array([0, 0, dz]), R_des=down_R(yaw))
    goto(qd, OPEN, maxn=20)
    p, _ = kin.fk(*rb(obs), rq(obs))
    print(' dz %.3f fk %s cube %s' % (dz, p.round(3), cube(obs).round(3)))

for _ in range(8):
    step(qd, CLOSE)
print('closed gripper pos', obs.get(R, 'pos_gripper'), 'cube', cube(obs).round(3))
qu, e = kin.ik(bx, by, br, rq(obs), c0 + np.array([0, 0, 0.30]), R_des=down_R(yaw))
goto(qu, CLOSE)
p, _ = kin.fk(*rb(obs), rq(obs))
print('lifted: fk', p.round(3), 'cube', cube(obs).round(3))
ok = cube(obs)[2] > 0.15
print('GRASP', 'SUCCESS' if ok else 'FAIL')

if ok:
    # calibration sweep: random reachable poses while holding cube
    rng = np.random.default_rng(SEED)
    for k in range(25):
        tgt = rng.uniform([0.45, -0.25, 0.2], [0.75, 0.25, 0.5])
        tgt = kin.base_T(bx, by, br)[:3, :3] @ tgt + np.array([bx, by, 0])
        zd = rng.normal(size=3) * 0.3 + np.array([0, 0, -1.0])
        zd /= np.linalg.norm(zd)
        qt, e = kin.ik(bx, by, br, rq(obs), tgt, z_des=zd)
        goto(qt, CLOSE, maxn=60, vmax=0.1)
        if cube(obs)[2] < 0.08:
            print('dropped at', k); break
        for _ in range(5):
            step(qt, CLOSE)
    print('final cube z', cube(obs)[2].round(3))
np.savez('pick_log_%d.npz' % SEED, log=np.array(log))
env.close()
