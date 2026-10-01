import sys, numpy as np, math
from env_client import make_env
import approach
from approach import GeneratedApproach
seed=int(sys.argv[1])
env=make_env(); ap=GeneratedApproach(env.action_space, env.observation_space, {})
obs,info=env.reset(seed=seed); ap.reset(obs,info)
orig=ap.get_action
def wrapped(state):
    a=orig(state)
    r=state.get_object_from_name("robot")
    x,y=float(state.get(r,"x")),float(state.get(r,"y"))
    nx,ny=x+a[0],y+a[1]
    print("  path",[(round(p,3),round(q,3)) for p,q in ap.path],"idx",ap.idx,"replans",ap._replans,
          "posefree",ap._pose_free(nx,ny,float(state.get(r,"theta"))+a[2]),"r",round(ap.radius,4),"gdir",ap.gdir)
    return a
ap.get_action=wrapped
for i in range(40):
    a=ap.get_action(obs)
    obs,rew,term,tr,_=env.step(a)
    r=obs.get_object_from_name("robot")
    print(i,"act",np.round(a,3),"pos",round(float(obs.get(r,"x")),4),round(float(obs.get(r,"y")),4),"th",round(float(obs.get(r,"theta")),2))
    if term: break
env.close()
