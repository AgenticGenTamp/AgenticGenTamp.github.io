from env_client import make_env
import numpy as np, sys
env = make_env()
R=None
F = ['pos_base_x','pos_base_y','pos_base_rot']+[f'pos_arm_joint{i}' for i in range(1,8)]+['pos_gripper']
def st(o): return np.array([float(o.get(o.get_object_from_name('robot'),f)) for f in F])
def run(seqs, label, idx=(0,1,2)):
    obs,_=env.reset(seed=0); s0=st(obs)
    print("==",label, "init", np.round(s0[list(idx)],4))
    for t,a in enumerate(seqs):
        obs,r,*_=env.step(np.array(a,float)); s=st(obs)
        print(t, np.round(a[:3],3), np.round(s[list(idx)],4), "d", np.round((s-s0)[list(idx)],4))
z=[0.]*11
def A(**k):
    a=list(z)
    for i,v in k.items(): a[int(i[1:])]=v
    return a
run([A(a0=-0.1)]+[z]*6, "base -x once")
run([A(a1=0.1)]+[z]*6, "base +y once")
run([A(a1=-0.1)]*5+[z]*4, "base -y x5")
run([A(a2=0.1)]+[z]*6, "rot once")
