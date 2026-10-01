"""Seek a cube grasp after changing one joint of the known box contact pose."""
import sys
import numpy as np
from env_client import make_env
from probe_grasp_structured import command, val

j = int(sys.argv[1]); delta = float(sys.argv[2]); seed = int(sys.argv[3])
target = "cube0"
qs = [np.array([.951305288,1.970876128,-1.493305392,-1.219847079,4.230377134,.474353059,6.00596593]),
      np.array([4.155821175,.426711007,-1.502109215,-1.900626345,1.884001343,.06090929,5.236141916]),
      np.array([3.870779777,1.703020948,-2.855671258,-.627929854,4.26973606,1.31947716,7.283258434])]
offs=[(-.799987478,-.347800459,2.104792961),(.126212298,-.083534270,-3.139703362),(-.428185141,.054943428,2.978131848)]
e=make_env();s,_=e.reset(seed=seed);tx=val(s,target,"pose_x");ty=val(s,target,"pose_y")
for q,o in zip(qs,offs):
 s=command(e,s,[tx+o[0],ty+o[1],o[2]],q,1.,35);s=command(e,s,[tx+o[0],ty+o[1],o[2]],q,-1.,2)
q=qs[-1].copy();q[j-1]+=delta;b=np.array([tx+offs[-1][0],ty+offs[-1][1]])
s=command(e,s,[b[0],b[1],offs[-1][2]],q,1.,20)
for iy,dy in enumerate(np.arange(-.24,.241,.02)):
 xs=np.arange(-.24,.241,.02)
 if iy%2:xs=xs[::-1]
 for dx in xs:
  bb=[b[0]+dx,b[1]+dy,offs[-1][2]];s=command(e,s,bb,q,1.,3);s=command(e,s,bb,q,-1.,2)
  if val(s,"robot","grasp_active")>.5:
   print("HIT",j,delta,dx,dy,"q",[val(s,"robot","joint_%d"%i) for i in range(1,8)],"off",val(s,"robot","pos_base_x")-tx,val(s,"robot","pos_base_y")-ty);e.close();raise SystemExit
print("MISS",j,delta);e.close()
