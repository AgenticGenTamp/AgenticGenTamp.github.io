from calib_util import *
import sys
axis=sys.argv[1] if len(sys.argv)>1 else 'x'
env=make_env(); obs,_=env.reset(seed=0)
cubes=['cube1','cube2','cube3','cube4']
by = -0.012 if axis=='x' else 0.0
obs,k=drive_base(env,obs,[0.55,by,np.pi],grip=1.0); b,q=rstate(obs); print('base',b.round(4))
Rd=down_R(0)
if axis=='x': a=np.array([0.25,0.0,0.30]); d=np.array([0.003,0,0])
else: a=np.array([0.55-0.008-0.12,0.08,0.30]); d=np.array([0,-0.003,0])  # world x=0.008 if mx=0
qc,_=ik(q,a,Rd,0); obs,_=joint_ctrl(env,obs,qc,K=1.3,steps=200,grip=1.0)
a[2]=0.2294+0.008; qc,_=ik(qc,a,Rd,0); obs,_=joint_ctrl(env,obs,qc,K=1.3,steps=200,grip=1.0)
c0={c:objpos(obs,c) for c in cubes}
for i in range(200):
    a=a+d; qc,_=ik(qc,a,Rd,0); obs,_=joint_ctrl(env,obs,qc,K=1.3,steps=5,grip=1.0)
    b,q=rstate(obs); tip=fk_arm(q,0)[0]
    mv={c:np.linalg.norm(objpos(obs,c)-c0[c]) for c in cubes}
    m=max(mv,key=mv.get)
    if mv[m]>0.001:
        print('moved',m,mv[m].round(4),'at arm tip',tip.round(4),'cube was',c0[m].round(4),'base',b.round(4)); break
