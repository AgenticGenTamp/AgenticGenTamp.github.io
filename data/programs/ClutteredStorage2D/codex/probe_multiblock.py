"""Run pinned multi-block episodes and summarize policy state transitions."""

import sys

from approach import GeneratedApproach
from env_client import make_env


def val(s, o, k):
    return float(s.get(o, k))


seed = int(sys.argv[1])
count = int(sys.argv[2])
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
env = make_env()
s, info = env.reset(seed=seed, options={"object_count": count})
p = GeneratedApproach(env.action_space, env.observation_space, {})
p.reset(s, info)
if len(sys.argv) > 4 and sys.argv[4] in ("push", "ram", "center"):
    p.to_clear = []
if len(sys.argv) > 4 and sys.argv[4] == "skipedge":
    p.to_clear = [n for n in p.to_clear
                  if .22 < val(s, s.get_object_from_name(n), "x") < 4.78]
if len(sys.argv) > 4 and sys.argv[4] == "clear":
    p.to_clear = [b.name for b in p._blocks(s) if p._inside(s, b)]
if len(sys.argv) > 4 and sys.argv[4] == "center":
    def center_slot(state, exclude):
        sh = state.get_objects(p.shelf_type)[0]
        return val(state, sh, "x1") + val(state, sh, "width1") / 2, val(state, sh, "y1") + .1
    p._choose_slot = center_slot
shelf = s.get_objects(p.shelf_type)[0]
print("shelf", {k: round(val(s, shelf, k), 3) for k in
                ("x", "y", "width", "height", "x1", "y1", "width1", "height1")})
print("initial", [(b.name, *(round(val(s, b, k), 3) for k in
                              ("x", "y", "theta", "width", "height")))
                  for b in p._blocks(s)])
old = None
for step in range(limit):
    a = p.get_action(s)
    if (len(sys.argv) > 4 and sys.argv[4] == "push" and p.stage == "insert" and
            val(s, s.get_object_from_name(p.target_name), "y") > 2.39):
        # Drop suction and use the end effector as a mechanical pusher.
        a = p._action(da=.04, vac=0.0)
    if (len(sys.argv) > 4 and sys.argv[4] == "ram" and p.stage == "insert" and
            val(s, s.get_object_from_name(p.target_name), "y") > 2.39):
        a = p._action(dy=.05, dt=.196, da=.1, vac=1.0)
    s, _, term, trunc, out = env.step(a)
    key = (p.stage, p.target_name, tuple(p.to_clear))
    if key != old or step % 100 == 99:
        blocks = [(b.name, round(val(s, b, "x"), 3), round(val(s, b, "y"), 3),
                   round(val(s, b, "theta"), 3), p._inside(s, b)) for b in p._blocks(s)]
        print(step + 1, key, blocks)
        old = key
    if term or trunc:
        print("END", step + 1, term, trunc, out)
        break
env.close()
