from env_client import make_env
import numpy as np, math, sys
def g(s,o,f): return float(s.get(o,f))
def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi

seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
offset=float(sys.argv[2]) if len(sys.argv)>2 else .78
xoff=float(sys.argv[3]) if len(sys.argv)>3 else 0
env=make_env(); s,info=env.reset(seed=seed); r=s.get_object_from_name('robot'); h=s.get_object_from_name('hook')
hx0,hy0=g(s,h,'x'),g(s,h,'y')
# Move to above hook, point downward, fully open. Hold target based initial hook to avoid chasing pushes.
for i in range(140):
 rx,ry,th=g(s,r,'x'),g(s,r,'y'),g(s,r,'theta')
 tx,ty=hx0+xoff,hy0+offset
 a=np.array([np.clip((tx-rx)*.5,-.03,.03),np.clip((ty-ry)*.5,-.03,.03),np.clip(wrap(-math.pi/2-th)*.5,-.098,.098),0,.015],np.float32)
 s,_,_,_,_=env.step(a)
# Close slowly and log
for i in range(30):
 s,_,_,_,_=env.step(np.array([0,0,0,0,-.015],np.float32))
 print(i,'R',*[round(g(s,r,f),4) for f in ('x','y','theta','arm_length','finger_gap')], 'H',*[round(g(s,h,f),4) for f in ('x','y','theta','held')])
 if g(s,h,'held')>.5: break
env.close()
