import numpy as np, sys
from env_client import make_env
import kin
np.set_printoptions(precision=3,suppress=True,linewidth=200)
env=make_env()
obs,_=env.reset(seed=int(sys.argv[1]) if len(sys.argv)>1 else 0)
QT=obs[96:103].copy()
def Rdown(psi):
    c,s=np.cos(psi),np.sin(psi)
    return np.array([[c,s,0],[s,-c,0],[0,0,-1.]])
def to_arm(pw,base):
    return kin.rotz(-base[2])@(np.array(pw)-np.array([base[0],base[1],0]))-kin.MOUNT
def goto(q_t,grip,obs,maxsteps=200,tol=0.003):
    global QT
    for i in range(maxsteps):
        a=np.zeros(11,np.float32); a[3:10]=np.clip(4*(q_t-QT),-0.1,0.1); a[10]=grip
        QT=QT+0.25*a[3:10].astype(np.float64)
        obs,r,te,tr,_=env.step(a)
        if np.max(np.abs(q_t-obs[96:103]))<tol: break
    return obs
def fkw(obs): return kin.fk_world(obs[93:96],obs[96:103])[0]
obj=int(sys.argv[3]) if len(sys.argv)>3 else 32
base=obs[93:96].copy(); q=obs[96:103].copy()
cp=obs[obj:obj+3].copy(); print('obj',cp,'base',base)
psi=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
Rt=kin.rotz(-base[2])@Rdown(psi)
pre,err=kin.ik_arm(to_arm(cp+[0,0,0.15],base),Rt,q,iters=300)
obs=goto(pre,0,obs,300)
print('fk pre',fkw(obs),'obj',obs[obj:obj+3], 'qerr',np.abs(obs[96:103]-pre).max())
g,err=kin.ik_arm(to_arm(cp+[0,0,0.0],base),Rt,pre)
obs=goto(g,0,obs,200)
print('fk g',fkw(obs),'obj',obs[obj:obj+3])
for i in range(15):
    a=np.zeros(11,np.float32); a[3:10]=0; a[10]=1; obs,*_=env.step(a)
print('closed obj',obs[obj:obj+7])
obs=goto(pre,1,obs,200)
print('lift fk',fkw(obs),'obj',obs[obj:obj+7])
