import sys, inspect, numpy as np, approach
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/BalanceBeam3D-o3-v0")
ap = approach.GeneratedApproach(env.action_space, env.observation_space, {})
om, ow = ap.move, ap.wait
log=[]
def wrapg(f, name):
    def g(*a, **kw):
        ln = inspect.stack()[1].lineno
        n=0
        for x in f(*a, **kw):
            n+=1; yield x
        log.append(f"{name}@{ln}:{n}")
    return g
ap.move = wrapg(om,'mv'); ap.wait = wrapg(ow,'w')
sd=int(sys.argv[1]) if len(sys.argv)>1 else 0
obs,info = env.reset(seed=sd); ap.reset(obs,info)
for t in range(1000):
    obs,r,term,*_ = env.step(ap.get_action(obs))
    if term: break
print(t+1, " ".join(log))
