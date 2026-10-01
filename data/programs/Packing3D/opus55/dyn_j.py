from env_client import make_env
import numpy as np
env = make_env()
def rs(obs):
    R=obs.get_object_from_name('robot')
    return np.array([float(obs.get(R,k)) for k in ['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{i}' for i in range(1,8)]+['finger_state','grasp_active']])
def stepa(**kw):
    a=np.zeros(11,np.float32)
    for k,v in kw.items(): a[int(k[1:])]=v
    return env.step(a)[0]
# table y extent
for y in [0.35,0.4,0.45,0.5,0.55]:
    env.reset(seed=0)
    for _ in range(int(round(y/0.05))): obs=stepa(a1=0.05)
    x0=rs(obs)[0]
    for _ in range(100): obs=stepa(a0=0.01)
    print("y",round(rs(obs)[1],3),"x stops at",round(rs(obs)[0],3))
# joint sweeps
for j in range(1,8):
    for d in [0.2,-0.2]:
        obs,_=env.reset(seed=0); s0=rs(obs); vals=[]
        for k in range(40):
            obs=stepa(**{f'a{j+2}':d}); vals.append(rs(obs)[j+2])
        v=np.array(vals); dif=np.diff(np.r_[s0[j+2],v]); dev=np.where(np.abs(dif-d)>1e-5)[0]
        print(f"j{j} d{d:+}: start {s0[j+2]:.3f} end {v[-1]:.3f} min {v.min():.3f} max {v.max():.3f} firstdev {dev[:1]} dif {np.round(dif[dev[:2]],3)}")
