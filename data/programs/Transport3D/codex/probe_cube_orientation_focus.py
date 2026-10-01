"""Dense verification of collision-reachable horizontal/vertical tool poses."""
import sys, math
import numpy as np
from env_client import make_env
from probe_grasp_structured import command,val

which=int(sys.argv[1]) if len(sys.argv)>1 else 0
qs=[
 [2.978067,-.283489,-2.176268,-1.017387,2.220226,.796092,5.877140], # tool x horizontal
 [2.419307,.700336,-2.155618,-1.001972,2.253750,.791872,1.621961], # opposite wrist roll
 [3.060388,.686246,-2.841432,-.617334,4.906317,.172242,5.445281], # tool vertical down
]
q=np.array(qs[which%len(qs)])
e=make_env();s,_=e.reset(seed=1);tx=val(s,'cube0','pose_x');ty=val(s,'cube0','pose_y')
# Full empirically collision-safe prefix, then smoothly transition to the new
# wrist orientation in small increments so collision projection cannot freeze it.
route=[
([.951305288,1.970876128,-1.493305392,-1.219847079,4.230377134,.474353059,6.00596593],[-.800,-.348,2.105]),
([4.155821175,.426711007,-1.502109215,-1.900626345,1.884001343,.06090929,5.236141916],[.126,-.084,-3.140]),
([4.170779777,1.703020948,-2.855671258,-.627929854,4.26973606,1.31947716,7.283258434],[-.428,.055,2.978])]
for qq,o in route:s=command(e,s,[tx+o[0],ty+o[1],o[2]],np.array(qq),1.,40)
base=[tx-.428,ty+.055,2.978]
q0=np.array([val(s,'robot','joint_%d'%j) for j in range(1,8)])
for f in np.linspace(0,1,9)[1:]:s=command(e,s,base,q0*(1-f)+q*f,1.,12)
real=np.array([val(s,'robot','joint_%d'%j) for j in range(1,8)])
print('target',q.tolist(),'real',real.tolist(),'maxerr',max(abs(q-real)),flush=True)
# 2.5 cm lattice: safely finer than the cube half-width.
ys=np.arange(-.36,.361,.025);xs=np.arange(-.72,.221,.025)
for iy,dy in enumerate(ys):
 row=xs if iy%2==0 else xs[::-1]
 for dx in row:
  b=[tx+dx,ty+dy,2.978]
  s=command(e,s,b,q,1.,4);s=command(e,s,b,q,-1.,1)
  if val(s,'robot','grasp_active')>.5:
   rb=[val(s,'robot',x) for x in ('pos_base_x','pos_base_y','pos_base_rot')]
   rq=[val(s,'robot','joint_%d'%j) for j in range(1,8)]
   print('FOCUS_CUBE_HIT which',which,'q',rq,'offset',[rb[0]-tx,rb[1]-ty,rb[2]],
         'tf',[val(s,'robot','grasp_tf_'+c) for c in 'xyz'],flush=True)
   e.close();raise SystemExit
print('FOCUS_NONE',which,flush=True);e.close()
