import numpy as np, sys
from env_client import make_env
import approach
from approach import GeneratedApproach
sd=int(sys.argv[1]) if len(sys.argv)>1 else 0
env=make_env(); obs,info=env.reset(seed=sd)
ap=GeneratedApproach(env.action_space, env.observation_space, {})
# monkeypatch to trace
orig_pp = GeneratedApproach._pick_and_place
def traced(self,name,slot,carry_z):
    print("PICKPLACE",name,np.round(slot,3))
    res = yield from orig_pp(self,name,slot,carry_z)
    print("  ->",res, "grasp",self._grasping(self._state), "partpos",np.round(self._ppos(self._state,name),4))
    return res
GeneratedApproach._pick_and_place=traced
origmove = GeneratedApproach._move
def tmove(self,*a,**k):
    yield from origmove(self,*a,**k)
    print("   move ->",self._last_result, np.round(self._linpos_world(self._state),3))
GeneratedApproach._move=tmove
ap.reset(obs,info)
for t in range(env.max_steps):
    a=ap.get_action(obs)
    obs,r,term,trunc,info=env.step(a)
    if term: break
print("steps",t+1,"term",term)
for n in sorted(x for x in obs.get_object_names() if x.startswith('part')):
    print(n, np.round(ap._ppos(obs,n),4))
env.close()
