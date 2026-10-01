import numpy as np, sys
from scoop_util import *
seed=int(sys.argv[1]); u=float(sys.argv[2]); OFF=0.043
r=R(seed); r.gripper(0.0,5)
st={}
def stop(r):
    lag=np.abs(r.q()-r.qi).max(); dm=np.linalg.norm(r.P('scoop_0')-st['P']); dq=np.abs(r.Q('scoop_0')-st['Q']).max()
    return lag>0.015 or dm>0.0005 or dq>0.003
row=[]
for v in np.arange(-0.065,0.0451,0.01):
    pw,th=local2world(r,u,v-OFF)   # finger at +tool x = +v
    Rw=rz(th)@RD
    lin(r,[pw[0],pw[1],0.51],Rw,vmax=0.01)
    st['P']=r.P('scoop_0'); st['Q']=r.Q('scoop_0')
    ok=lin(r,[pw[0],pw[1],0.448],Rw,vmax=0.0015,stop=stop)
    z=r.fk()[0][2]+0.0125-0.46
    row.append('%.3f%s'%(z,' ' if ok else '*'))
    r.qi=r.q().copy()
    lin(r,[pw[0],pw[1],0.51],Rw,vmax=0.01)
print('u %+.3f '%u,' '.join(row),flush=True)
