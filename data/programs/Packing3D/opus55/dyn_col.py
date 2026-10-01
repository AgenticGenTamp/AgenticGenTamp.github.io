from env_client import make_env
import numpy as np
env = make_env()
def rs(obs):
    R=obs.get_object_from_name('robot')
    return np.array([float(obs.get(R,k)) for k in ['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{i}' for i in range(1,8)]+['finger_state','grasp_active']])
def run(idx, d, n=40, seed=0, pre=None):
    obs,_=env.reset(seed=seed)
    if pre:
        for a in pre: obs,*_=env.step(a)
    s0=rs(obs); traj=[]
    for k in range(n):
        a=np.zeros(11,np.float32); a[idx]=d
        obs,r,te,tr,info=env.step(a); s=rs(obs); traj.append(s[idx] if idx<10 else s[10])
        if te or tr: print("term/trunc",te,tr); break
    tr_=np.array(traj); dif=np.diff(np.r_[s0[idx] if idx<10 else s0[10],tr_])
    stuck=np.where(np.abs(dif-d)>1e-6)[0]
    print(f"idx{idx} d{d}: start {s0[idx]:.4f} end {tr_[-1]:.4f} first-deviation-step {stuck[:1]} diffs@dev {np.round(dif[stuck[:3]],4)} other-changed {np.abs(rs(obs)-s0)[[i for i in range(10) if i!=idx]].max():.4f}")
for d in [0.2,0.05,0.01]: run(0,d,n=30)
for d in [-0.2,-0.05]: run(0,d,n=60)
for d in [0.2,-0.2,0.05]: run(1,d,n=60)
for d in [0.2,-0.2]: run(2,d,n=40)
print("---combined")
obs,_=env.reset(seed=0)
a=np.zeros(11,np.float32); a[0]=0.2; a[3]=0.1
obs,*_=env.step(a); print("base+x0.2 & dq1=0.1 ->", np.round(rs(obs)[:4],4))
obs,_=env.reset(seed=0)
a=np.zeros(11,np.float32); a[0]=0.06; a[3]=0.1
obs,*_=env.step(a); print("base+x0.06 & dq1=0.1 ->", np.round(rs(obs)[:4],4))
for y in [0.3,0.6,1.0,2.0]:
    pre=[]; n=int(round(y/0.2)) if y>=0.2 else 0
    a=np.zeros(11,np.float32); a[1]=y/ max(n,1)
    pre=[a]*max(n,1)
    print("at y",y,end=' '); run(0,0.01,n=200,pre=pre)
