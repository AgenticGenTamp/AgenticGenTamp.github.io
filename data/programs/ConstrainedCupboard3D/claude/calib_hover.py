"""Detect arm mount offset by lowering the predicted fingertip onto a rod."""
import sys, numpy as np, kin
from env_client import make_env

MX, MY, MZ = (float(x) for x in sys.argv[1:4])  # mount offset in base frame
TOOL = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15

def robot_q(o):
    r = o.get_object_from_name('robot'); f = o.type_features[r.type]
    d = o.data[r]
    return np.array([d[f.index('pos_arm_joint%d'%(i+1))] for i in range(7)]), \
           np.array([d[f.index('pos_base_x')], d[f.index('pos_base_y')], d[f.index('pos_base_rot')]])

def goto(env, o, q_des, gripper, nsteps=60, watch=None):
    for i in range(nsteps):
        q, b = robot_q(o)
        a = np.zeros(11, dtype=np.float32)
        a[3:10] = np.clip((q_des - q) * 1.0, -0.1, 0.1)
        a[10] = gripper
        o, rew, t, tr, inf = env.step(a)
        if watch is not None:
            w = o.get_object_from_name(watch)
            if np.linalg.norm(o.data[w][7:13]) > 0.02:
                return o, True, i
    return o, False, nsteps

env = make_env()
o, info = env.reset(seed=0)
q, b = robot_q(o)
print("base", b)
# nearest rod
rods = [n for n in o.get_object_names() if n.startswith('cuboid')]
best = min(rods, key=lambda n: np.hypot(o.data[o.get_object_from_name(n)][0]-b[0],
                                        o.data[o.get_object_from_name(n)][1]-b[1]))
rd = o.data[o.get_object_from_name(best)]
print("rod", best, rd[:7])
# rod pos in arm frame
c, s = np.cos(b[2]), np.sin(b[2])
dx, dy = rd[0]-b[0], rd[1]-b[1]
px = c*dx + s*dy - MX
py = -s*dx + c*dy - MY
print("rod in arm frame xy", px, py)
# gripper pointing down
R = np.array([[1,0,0],[0,-1,0],[0,0,-1.0]])
for h in [0.30, 0.20, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.0, -0.02, -0.04]:
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = [px, py, rd[2]-MZ+h]
    qd, ep, er = kin.ik(T, q, tool_z=TOOL, q_lim=None)
    if ep > 0.01:
        print("h", h, "IK FAIL", ep, er); continue
    o, hit, i = goto(env, o, qd, 1.0, 40, watch=best)
    q, b = robot_q(o)
    rd2 = o.data[o.get_object_from_name(best)]
    print("h=%.3f iters=%d hit=%s poserr=%.4f rod=%s" % (h, i, hit, np.linalg.norm(kin.arm_fk(q,TOOL)[:3,3]-T[:3,3]), np.round(rd2[:3],4)))
    if hit: break
env.close()
