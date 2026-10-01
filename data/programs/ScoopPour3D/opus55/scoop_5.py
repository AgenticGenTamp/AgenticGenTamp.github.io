import numpy as np, sys
from scoop_util import *
seed=int(sys.argv[1]); u=float(sys.argv[2]); v=float(sys.argv[3]); dy=float(sys.argv[4]); zt=float(sys.argv[5])
tag=' '.join(sys.argv[1:])
r=R(seed); r.gripper(0.0,5)
P0=r.P('scoop_0'); Q0=r.Q('scoop_0')
pw,th=local2world(r,u,v); Rw=rz(th+dy)@RD
lin(r,[pw[0],pw[1],0.53],Rw,vmax=0.01)
lin(r,[pw[0],pw[1],zt],Rw,vmax=0.004)
mv=np.linalg.norm(r.P('scoop_0')-P0)
r.gripper(1.0,20); g=r.grip()
r.qi=r.q().copy()
lin(r,[pw[0],pw[1],0.58],Rw,vmax=0.005)
for k in range(10): r.step(dq=np.zeros(7))
P=r.P('scoop_0'); Q=r.Q('scoop_0')
tp,TR=r.fk()
# scoop pose in tool frame
Rs=kin.quat2mat(Q); rel=TR.T@(P-tp); Rrel=TR.T@Rs
print(tag,'| push %.4f grip %.3f | scoop z %.3f lifted %s | rel_pos'%(mv,g,P[2],P[2]>0.52),rel.round(4),'Rrel',Rrel.round(2).tolist(),flush=True)
if len(sys.argv)>6:
    import pickle
    img=r.env.render(); pickle.dump(img,open('scoop_img.pkl','wb')); print(type(img), getattr(img,'shape',None))
