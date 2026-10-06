import sys, math, numpy as np
from robocode.environments.rovers_variable_count_env import RoversVariableCountEnv
import approach as A
seed=int(sys.argv[1]); R=int(sys.argv[2])
env = RoversVariableCountEnv(design_counts=[1,2,3,4], eval_counts=[1,2,3,4])
state, info = env.reset(seed=seed)
a = A.GeneratedApproach(env.action_space, env.observation_space, {})
a.reset(state, info)
for k in range(40):
    t=a.t[R]; plan=a.plans[R]
    rv=a._parse(state)[0][R]
    act=a.get_action(state)
    if t < len(plan) and plan[t][1] is not None:
        (px,py),op=plan[t]
        m=a._op_cone(R,px,py,op)
        bins=np.where(m)[0] if m is not None else []
        newth=rv['theta']+act[4*R+2]
        print("t",t,"op",op,"at",round(px,2),round(py,2),"th_after",round(float(newth),2),"cone",[round(float(A.BIN_C[b]),2) for b in bins][::4])
    state,_,term,_,_=env.step(act)
    if term: print("done",k+1); break
o=a._parse(state)[0][1-R]; print("other rover", o['x'],o['y'])
