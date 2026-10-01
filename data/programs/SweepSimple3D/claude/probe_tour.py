import numpy as np, fk, ctrl, json, sys
from env_client import make_env
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
env=make_env(); obs,info=env.reset(seed=seed)
def g(n,f): return float(obs.get(obs.get_object_from_name(n),f))
def J(): return np.array([g("robot","pos_arm_joint%d"%i) for i in range(1,8)])
def B(): return np.array([g("robot","pos_base_x"),g("robot","pos_base_y"),g("robot","pos_base_rot")])
names=[n for n in obs.get_object_names() if n.startswith("cube_")]
def CS(): return {n:(round(g(n,"x"),3),round(g(n,"y"),3)) for n in names}
rews=[]
def step(tq,tb=None,grip=0.0):
    global obs
    a=ctrl.action(J(),tq,B(),tb,grip); obs,rew,t,tr,i2=env.step(a)
    rews.append(rew)
    if abs(rew+1.0)>1e-6:
        print("REWARD CHANGE",rew,"cubes",CS(),"base",np.round(B(),3),flush=True)
    return rew,t
yaw=B()[2]
q=ctrl.ik_local([0.45,0.0,-0.02],J())
for _ in range(140): step(q)
print("jerr",round(float(np.max(np.abs(J()-q))),3),"cubes",CS(),flush=True)
# raster the room with the arm down: base sweeps lines
xs=np.arange(0.7,2.5,0.18)
n=0; term=False
for i,x in enumerate(xs):
    ys=[0.15,2.3] if i%2==0 else [2.3,0.15]
    for tgt in ys:
        while True:
            b=B()
            tb=np.array([x, tgt, yaw])
            r,t=step(q,tb); n+=1
            if t: term=True; break
            if abs(b[0]-x)<0.02 and abs(b[1]-tgt)<0.04: break
            if n>950: break
        if term or n>950: break
    if term or n>950: break
print("steps",n,"terminated",term,"cubes",CS())
print("unique rewards",sorted(set([round(r,4) for r in rews])))
env.close()
