import numpy as np, sys
from scoop_util import *
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
r=R(seed); r.gripper(1.0,10)
print('scoop',r.P('scoop_0'),yaw(r.Q('scoop_0')))
st={}
def stop(r):
    lag=np.abs(r.q()-r.qi).max(); dm=np.linalg.norm(r.P('scoop_0')-st['P'])
    return lag>0.02 or dm>0.001
for u in np.arange(-0.07,0.0701,0.014):
  row=[]
  for v in np.arange(-0.049,0.0491,0.014):
    pw,th=local2world(r,u,v)
    lin(r,[pw[0],pw[1],0.515],vmax=0.01)
    st['P']=r.P('scoop_0')
    ok=lin(r,[pw[0],pw[1],0.463],vmax=0.003,stop=stop)
    z=r.fk()[0][2]-0.001
    mv=np.linalg.norm(r.P('scoop_0')-st['P'])
    row.append('%.3f%s'%(z-0.46,'*' if mv>0.001 else ' '))
    r.qi=r.q().copy()
    lin(r,[pw[0],pw[1],0.515],vmax=0.01)
  print('u %+.3f '%u,' '.join(row),flush=True)
print('scoop',r.P('scoop_0'),yaw(r.Q('scoop_0')))
