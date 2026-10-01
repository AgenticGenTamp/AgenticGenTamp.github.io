from env_client import make_env
import numpy as np, sys, json
seed=int(sys.argv[1]); wps=json.loads(sys.argv[2]); step=float(sys.argv[3]) if len(sys.argv)>3 else 0.02
env = make_env()
import os; oc=os.environ.get("OC"); obs, info = env.reset(seed=seed, options={"object_count":int(oc)} if oc else None)
R = env.observation_space.get_type('mujoco_tidybot_robot')
def rs(o):
    r=o.get_objects(R)[0]; return np.array([o.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
t=0
for k,(tx,ty) in enumerate(wps):
    lim = 0.1 if k==0 else step
    for i in range(400):
        p=rs(obs); d=np.array([tx,ty])-p[:2]
        if np.linalg.norm(d)<0.01: break
        a=np.zeros(11,dtype=np.float32); a[:2]=np.clip(d,-lim,lim)
        obs,rew,term,trunc,info=env.step(a); t+=1
        if term: print(sys.argv[2][:40],'TERM',rs(obs).round(3), 'prev',p.round(3)); sys.exit()
print(sys.argv[2][:40],'none',rs(obs).round(2))
