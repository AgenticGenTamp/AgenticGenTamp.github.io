from env_client import make_env
from kin import *; from helpers import *
import sys
MX,MZ=0.12,0.425; L=0.16
yaw_g=float(sys.argv[1]) if len(sys.argv)>1 else 0.0
env=make_env(); obs,info=env.reset(seed=1)
s=rstate(obs); q=s[3:10]
c=cubes(obs)['cube1']
# cube in base frame (base yaw 0)
cb=np.array([c[0]-s[0]-MX, c[1]-s[1], c[2]-MZ])
Rd=np.array([[np.cos(yaw_g),-np.sin(yaw_g),0],[np.sin(yaw_g),np.cos(yaw_g),0],[0,0,1.]])@np.array([[0,1,0],[1,0,0],[0,0,-1.]])
for dz,g in [(0.15,0),(0.0,0),(0.0,1),(0.2,1)]:
    p=cb+np.array([0,0,dz+L-0.02])
    qd,_,_=ik(p,Rd,q)
    if dz==0 and g==1:
        for _ in range(15):
            a=np.zeros(11,np.float32); a[10]=1; obs,*_=env.step(a)
    obs,t,ok=drive(env,obs,np.concatenate([s[:3],qd]),g,steps=150,tol=0.005)
    q=rstate(obs)[3:10]
    T=fk_arm(q); 
    print(dz,g,t,ok,'fk',T[:3,3].round(3),'cube_armframe',(cubes(obs)['cube1'][:3]-np.array([s[0]+MX,s[1],MZ])).round(3), 'grip',rstate(obs)[10].round(3))
T=fk_arm(q); cw=cubes(obs)['cube1'][:3]-np.array([s[0],s[1],0])
print('cube in base frame',cw.round(4),'interface arm',T[:3,3].round(4),'R',T[:3,:3].round(2).tolist())
print('cube quat', cubes(obs)['cube1'][3:].round(3))
