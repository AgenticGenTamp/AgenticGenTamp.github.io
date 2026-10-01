"""Sweep scaled Cartesian-down adjustment of the reproducible box pose."""
import sys
import numpy as np

from env_client import make_env
from probe_grasp_structured import command, val


scale = float(sys.argv[1])
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
target = "cube0"
qs = [
    np.array([.951305288, 1.970876128, -1.493305392, -1.219847079,
              4.230377134, .474353059, 6.00596593]),
    np.array([4.155821175, .426711007, -1.502109215, -1.900626345,
              1.884001343, .06090929, 5.236141916]),
    np.array([3.870779777, 1.703020948, -2.855671258, -.627929854,
              4.26973606, 1.31947716, 7.283258434]),
]
offs = [(-.799987478, -.347800459, 2.104792961),
        (.126212298, -.083534270, -3.139703362),
        (-.428185141, .054943428, 2.978131848)]
# One previously measured inverse local-lift increment (about 4 cm down).
dq = np.array([.008, 0., -.029, .045, .003, -.107, -.062])
db = np.array([-.043, -.024])

env = make_env(); state, _ = env.reset(seed=seed)
tx, ty = val(state, target, "pose_x"), val(state, target, "pose_y")
for q, off in zip(qs, offs):
    state = command(env, state, [tx+off[0], ty+off[1], off[2]], q, 1., 35)
    state = command(env, state, [tx+off[0], ty+off[1], off[2]], q, -1., 2)
q = qs[-1] + scale*dq
b0 = np.array([tx+offs[-1][0], ty+offs[-1][1]]) + scale*db
state = command(env, state, [b0[0], b0[1], offs[-1][2]], q, 1., 20)
actual_q = [val(state, "robot", "joint_%d" % i) for i in range(1, 8)]
# Fine translation search accounts for local-Jacobian and grasp-offset error.
for iy, oy in enumerate(np.arange(-.10, .1001, .01)):
    xs = np.arange(-.10, .1001, .01)
    if iy % 2: xs = xs[::-1]
    for ox in xs:
        base = [b0[0]+ox, b0[1]+oy, offs[-1][2]]
        state = command(env, state, base, q, 1., 3)
        state = command(env, state, base, q, -1., 2)
        if val(state, "robot", "grasp_active") > .5:
            print("HIT scale", scale, "seed", seed, "adjust", ox, oy,
                  "q", actual_q, "baseoff", val(state,"robot","pos_base_x")-tx,
                  val(state,"robot","pos_base_y")-ty, flush=True)
            env.close(); raise SystemExit
print("MISS scale", scale, "actual_q", actual_q, flush=True)
env.close()
