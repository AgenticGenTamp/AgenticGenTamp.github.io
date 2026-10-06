import sys
from exp_squeeze2 import get_env, cases, set_block, wall_gaps, np
from exp_common import blocks
from exp_primitive import WallUnjam, _gripper_parts
lab = int(sys.argv[1]); cseed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
cs = cases(30 if cseed else 16, cseed)
n, seed, name, mod = [c for c in cs if c[1] == lab][0]
env = get_env(n); state, _ = env.reset(seed=seed, options={"object_count": n})
if mod: state = set_block(env, state, name, **mod)
P = WallUnjam(env.action_space, name); last = None
for T in range(1500):
    a = P.step(state)
    if a is None: break
    if P.phase != last or P.phase in ("descend", "push"):
        r, b = P._parse(state)
        print(T, P.ci, P.phase, "r=(%.3f %.3f %.3f L%.2f g%.2f)" % (r['x'], r['y'], r['theta'], r['arm_joint'], r['finger_gap']),
              "b=(%.3f %.3f th%.3f) gap%.3f bx%.3f yf%.3f" % (b['x'], b['y'], b['theta'], P.wall_gap(b), P.bx, P.y_final))
        last = P.phase
    state, *_ = env.step(a)
