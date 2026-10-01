from env_client import make_env
import numpy as np, sys
seed=int(sys.argv[1])
env = make_env()
obs, info = env.reset(seed=seed)
R = env.observation_space.get_type('mujoco_tidybot_robot')
M = env.observation_space.get_type('mujoco_movable_object')
def rs(o):
    r=o.get_objects(R)[0]; return np.array([o.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
def ch(o): return {c.name:[round(o.get(c,f),2) for f in ['x','y','z']] for c in o.get_objects(M)}
print(ch(obs), rs(obs))
t=0
def goto(tx,ty,maxs=200):
    global obs,t
    for i in range(maxs):
        p=rs(obs); d=np.array([tx,ty])-p[:2]
        if np.linalg.norm(d)<0.05: return None
        a=np.zeros(11,dtype=np.float32); a[:2]=np.clip(d,-0.1,0.1)
        obs,rew,term,trunc,info=env.step(a); t+=1
        if rew!=-1.0 or term: print('t',t,'rew',rew,term,rs(obs).round(2),ch(obs)); return True
    return 'stuck'
x0,y0=[float(v) for v in sys.argv[2:4]]
for yy in np.arange(y0-1.5,y0+1.6,0.3):
    if goto(x0-1.5,yy): break
    if t%50==0: print(t, rs(obs).round(2), ch(obs))
    if goto(x0+1.5,yy): break
    print(t, rs(obs).round(2), ch(obs))
