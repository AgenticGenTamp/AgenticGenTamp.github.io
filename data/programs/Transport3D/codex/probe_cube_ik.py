"""Test model-derived 75 mm downward IK from the reproducible box pose."""
import numpy as np
from env_client import make_env
from probe_grasp_structured import command, val

qs=[np.array([.951305288,1.970876128,-1.493305392,-1.219847079,4.230377134,.474353059,6.00596593]),np.array([4.155821175,.426711007,-1.502109215,-1.900626345,1.884001343,.06090929,5.236141916]),np.array([3.870779777,1.703020948,-2.855671258,-.627929854,4.26973606,1.31947716,7.283258434])]
offs=[(-.799987478,-.347800459,2.104792961),(.126212298,-.083534270,-3.139703362),(-.428185141,.054943428,2.978131848)]
q=np.array([3.74229097,1.57453213,-3.37580215,-.54634759,3.82189692,.98145015,7.24122776])
e=make_env();s,_=e.reset(seed=1);tx=val(s,'cube0','pose_x');ty=val(s,'cube0','pose_y')
for qq,o in zip(qs,offs):s=command(e,s,[tx+o[0],ty+o[1],o[2]],qq,1.,35);s=command(e,s,[tx+o[0],ty+o[1],o[2]],qq,-1.,2)
b=np.array([tx+offs[-1][0],ty+offs[-1][1]]);s=command(e,s,[b[0],b[1],offs[-1][2]],q,1.,30)
for iy,dy in enumerate(np.arange(-.2,.201,.01)):
 xs=np.arange(-.2,.201,.01)
 if iy%2:xs=xs[::-1]
 for dx in xs:
  bb=[b[0]+dx,b[1]+dy,offs[-1][2]];s=command(e,s,bb,q,1.,3);s=command(e,s,bb,q,-1.,2)
  if val(s,'robot','grasp_active')>.5:print('HIT',dx,dy,'q',[val(s,'robot','joint_%d'%i) for i in range(1,8)],'off',val(s,'robot','pos_base_x')-tx,val(s,'robot','pos_base_y')-ty);e.close();raise SystemExit
print('MISS q',[val(s,'robot','joint_%d'%i) for i in range(1,8)]);e.close()
