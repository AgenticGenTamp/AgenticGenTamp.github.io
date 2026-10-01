from env_client import make_env
import numpy as np
from topdown_grasp_probe import get,rob,xyz,act,run

EDGE=np.array([0.,1.3,np.pi,-1.7,0.,1.,0.])
HIGH=np.array([0.,.9,np.pi,-1.7,0.,1.,np.pi/2])

def trial(lowq,yoff=0):
 e=make_env();s,info=e.reset(seed=0,options={"object_count":4});home=rob(s)[3:10];c="cube1"; cubes=[n for n in s.get_object_names() if n.startswith("cube")]
 # Left-side setup for short tangential isolation sweep.
 s=run(e,s,45,[1,.8,np.pi],home,1);s=run(e,s,55,[-1,.8,np.pi],home,1);s=run(e,s,45,[-1,.8,0],home,1)
 y=float(xyz(s,c)[1]+.042);s=run(e,s,45,[-1,y,0],home,1);s=run(e,s,130,[-1,y,0],EDGE,1)
 all0={n:xyz(s,n).copy() for n in cubes};hit=None
 for k in range(40):
  a=act(s,[-.8,y,0],EDGE,1);a[0]=min(a[0],.012);s,*_=e.step(a)
  if max(np.linalg.norm(xyz(s,n)-v) for n,v in all0.items())>.001:hit=rob(s)[:3].copy();break
 qt=EDGE.copy();qt[0]=.5
 for k in range(80):
  s,*_=e.step(act(s,hit,qt,1))
  if xyz(s,c)[1]<-.085:break
 isolated=xyz(s,c).copy();print("ISOLATED",lowq,"k",k,"q1",round(rob(s)[3],3),"cube",np.round(isolated,4).tolist())
 # Pull away, raise/configure open broad jaws, center above isolated cube.
 s=run(e,s,25,[-1.08,rob(s)[1],0],qt,1)
 bx=float(xyz(s,c)[0]-.91); y=float(xyz(s,c)[1]+yoff)
 s=run(e,s,100,[-1.08,y,0],HIGH,1);s=run(e,s,30,[bx,y,0],HIGH,1)
 low=HIGH.copy();low[1]=lowq
 before=xyz(s,c).copy();s=run(e,s,60,[bx,y,0],low,1);down=xyz(s,c).copy();s=run(e,s,20,[bx,y,0],low,0);closed=xyz(s,c).copy()
 maxz=closed[2]
 for k in range(70):
  s,*_=e.step(act(s,[bx,y,0],HIGH,0));maxz=max(maxz,xyz(s,c)[2])
 after=xyz(s,c)
 print("RESULT",lowq,"yoff",yoff,"bx",round(bx,4),"before",np.round(before,4).tolist(),"down",np.round(down,4).tolist(),"closed",np.round(closed,4).tolist(),"after",np.round(after,4).tolist(),"maxz",round(maxz,4),"robot",np.round(rob(s),3).tolist())
 e.close()

if __name__=="__main__":
 for yo in [-.02,-.04]:trial(1.30,yo)
