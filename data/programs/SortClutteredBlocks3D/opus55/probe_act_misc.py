from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
R = obs.get_object_from_name('robot')
J=['pos_arm_joint%d'%i for i in range(1,8)]
def js(o): return np.array([float(o.get(R,f)) for f in J])
A = lambda: np.zeros(11, dtype=np.float32)
# sag: move j2 by some then hold zero 100 steps
a=A(); a[4]=0.1
for i in range(20): obs,*_=env.step(a)
j0=js(obs)
for i in range(100): obs,*_=env.step(A())
print('hold 100 zero steps drift', np.round(js(obs)-j0,4))
# j4 limits
for sgn in [1,-1]:
    obs,_=env.reset(seed=0); a=A(); a[6]=0.1*sgn
    tr=[]
    for i in range(60):
        obs,*_=env.step(a)
        if i%10==9: tr.append(round(js(obs)[3],3))
    print('j4 sign',sgn,tr)
# base toward -x until stuck
obs,_=env.reset(seed=0); a=A(); a[0]=-0.1
prev=None
for i in range(20):
    obs,r,te,tr,info=env.step(a)
    x=float(obs.get(R,'pos_base_x'))
    print(i, round(x,4), r, te, tr) if i%2==1 or (prev is not None and abs(prev-x)<0.01) else None
    prev=x
cube=obs.get_object_from_name('cube1'); print('cube1', [round(float(obs.get(cube,f)),3) for f in ['x','y','z']])
T=obs.get_object_from_name('table_1'); print('table', [round(float(obs.get(T,f)),3) for f in ['x','y','z']])
# long episode for truncation
obs,_=env.reset(seed=0); n=0
while True:
    obs,r,te,tr,info=env.step(A()); n+=1
    if te or tr: print('ended at',n,te,tr,r,info); break
    if n>1100: print('no end by 1100'); break
