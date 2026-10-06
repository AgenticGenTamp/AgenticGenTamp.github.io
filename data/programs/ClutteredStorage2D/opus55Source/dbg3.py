import sys, time, math
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
import approach
from approach import *
n, seed, stop = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.clutteredstorage2d:ClutteredStorage2DEnv",count_kwarg="num_blocks",count_object_prefix="block",bilevel_env_name="clutteredstorage2d",design_counts=[n],eval_counts=[n])
s, info = env.reset(seed=seed, options={'object_count': n})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(s, info)
for step in range(stop):
    a = ap.get_action(s)
    s, r, term, trunc, info = env.step(a)
w = ap.world; w.update(s)
print("q", w.q, "vac", w.vac, "x1", w.x1, "w1", w.w1, "ncols", w.ncols)
for nme in sorted(w.blocks): print(nme, [round(v,3) for v in w.bpose[nme]], "center", [round(v,3) for v in w.block_center(nme)], "inside", w.inside_slot(nme))
print("status", ap._column_status())
print("tasks", ap._choose_task())
ck = Checker(w, margin=0.006)
print("start collides", ck.collides(w.q), ck.collides((w.q[0],w.q[1],w.q[2],0.2)), Checker(w,margin=0.0).collides(w.q))
n_ = 'block2'
ck_free = Checker(w, margin=0.006); ck_grasp = Checker(w, margin=0.004)
cands = ap._grasp_candidates(n_)
from collections import Counter
c = Counter()
for qg, off, side in cands:
    if ck_grasp.collides(qg): c['grasp_coll']+=1; continue
    if not ap._suction_only(qg, n_): c['suction']+=1; continue
    qa = (qg[0], qg[1], qg[2], ARM_MIN)
    if not ck_grasp.path_ok(densify(qa, qg)): c['approach']+=1; continue
    if ck_free.collides(qa): c['qa']+=1; continue
    rel = se2_mul(se2_inv(tip_pose(qg)), w.bpose[n_])
    tr = ap._plan_transport(qg, (n_, rel), 1, 1, time_limit=1.0)
    if tr is None: c['transport']+=1; continue
    c['ok']+=1
print(c)
