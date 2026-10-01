import numpy as np
from env_client import make_env
from ik import ik
from fk import fk

JNAMES=[f'joint_{i}' for i in range(1,8)]
def getq(obs):
    r=obs.get_object_from_name('robot')
    return np.array([obs.get(r,n) for n in JNAMES]), r

env=make_env()
obs,info=env.reset(seed=0)
q,r = getq(obs)
p0 = obs.get_object_from_name('part0')
px,py,pz = [obs.get(p0,f) for f in ['pose_x','pose_y','pose_z']]
bx,by = obs.get(r,'pos_base_x'), obs.get(r,'pos_base_y')
print("part", px,py,pz, "base",bx,by)
# target in base frame
Rdown = np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)
tgt = np.array([px-bx, py-by, 0.30])
qd, M = ik(q, tgt, Rdown)
print("ik result pos", M[:3,3], "target", tgt)
print("qd", qd)
# move there in steps
def move_to(env, obs, qd, maxsteps=60):
    for k in range(maxsteps):
        q,r = getq(obs)
        d = qd - q
        if np.max(np.abs(d))<1e-3: break
        a=np.zeros(11); a[3:10]=np.clip(d,-0.2,0.2)
        obs,rew,term,trunc,info = env.step(a)
    return obs
obs = move_to(env,obs,qd)
q,_=getq(obs)
print("reached q", q, "err", np.max(np.abs(q-qd)))
print("fk pos", fk(q)[:3,3])
# now descend and try close at each height
for z in np.arange(0.30, -0.05, -0.02):
    tgt = np.array([px-bx, py-by, z])
    qd,M = ik(q, tgt, Rdown)
    obs = move_to(env,obs,qd)
    q,_=getq(obs)
    a=np.zeros(11); a[10]=-1.0
    obs,*_=env.step(a)
    rr=obs.get_object_from_name('robot')
    print(f"z={z:.3f} fkpos={np.round(fk(q)[:3,3],3)} qerr={np.max(np.abs(q-qd)):.3f} grasp={obs.get(rr,'grasp_active')} finger={obs.get(rr,'finger_state')} partz={obs.get(p0,'pose_z')}")
    a=np.zeros(11); a[10]=1.0
    obs,*_=env.step(a)
env.close()
