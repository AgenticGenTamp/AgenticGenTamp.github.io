from env_client import make_env
import numpy as np, sys, json
seed=int(sys.argv[1]); wps=json.loads(sys.argv[2])
env = make_env()
obs, info = env.reset(seed=seed)
R = env.observation_space.get_type('mujoco_tidybot_robot')
M = env.observation_space.get_type('mujoco_movable_object')
def rs(o):
    r=o.get_objects(R)[0]; return np.array([o.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
def ch(o): return {c.name:[round(o.get(c,f),2) for f in ['x','y']] for c in o.get_objects(M)}
print(ch(obs), rs(obs).round(2))
t=0
for tx,ty in wps:
    for i in range(200):
        p=rs(obs); d=np.array([tx,ty])-p[:2]
        if np.linalg.norm(d)<0.03: break
        a=np.zeros(11,dtype=np.float32); a[:2]=np.clip(d,-0.1,0.1)
        obs,rew,term,trunc,info=env.step(a); t+=1
        if term: print('TERM t',t,rew,rs(obs).round(2),ch(obs)); sys.exit()
    print('at',(tx,ty),'t',t,rs(obs).round(2),ch(obs))
