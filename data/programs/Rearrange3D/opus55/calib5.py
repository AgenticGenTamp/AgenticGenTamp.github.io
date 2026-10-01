import numpy as np, sys, json
exec(open('calib1.py').read().split("obs=goto(pre,0,obs,300)")[0])
tgt=[0.2,0.0,0.8]
g,err=kin.ik_arm(to_arm(np.array(tgt),base),Rt,obs[96:103],iters=300)
print('start q',obs[96:103],'g',g)
for i in range(60):
    q=obs[96:103]
    a=np.zeros(11,np.float32); a[3:10]=np.clip(4*(g-q),-0.1,0.1)
    obs,*_=env.step(a)
    if i%5==0: print(i,'a',a[3:10],'q-g',obs[96:103]-g)
