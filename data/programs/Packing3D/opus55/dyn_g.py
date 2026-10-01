from env_client import make_env
import numpy as np
env = make_env()
def rs(obs):
    R=obs.get_object_from_name('robot')
    return np.array([float(obs.get(R,k)) for k in ['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{i}' for i in range(1,8)]+['finger_state','grasp_active']])
def stepa(**kw):
    a=np.zeros(11,np.float32)
    for k,v in kw.items(): a[int(k[1:])]=v
    return env.step(a)
obs,_=env.reset(seed=0)
fs=[]
for g in [-1]*8+[0]*3+[1]*8+[-0.3]*2+[-0.6]*3+[0.6]*3:
    obs,r,te,tr,info=stepa(a10=g); s=rs(obs); fs.append((g,round(s[10],3),s[11]))
print(fs); print("reward sample",r,"info",info)
# small step joint collision boundaries
for j,d in [(2,0.01),(4,-0.01),(6,-0.01)]:
    obs,_=env.reset(seed=0); s0=rs(obs)
    for k in range(200):
        obs,*_=stepa(**{f'a{j+2}':d})
    print(f"j{j} d{d}: {s0[j+2]:.3f} -> {rs(obs)[j+2]:.3f}")
# clipping check near limit
obs,_=env.reset(seed=0)
for k in range(10): obs,*_=stepa(a5=-0.2)
print("j2 after 10x-0.2", rs(obs)[4])
