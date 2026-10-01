from env_client import make_env
import numpy as np, math, sys

def g(s,o,f): return float(s.get(o,f))
def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
env=make_env(); s,info=env.reset(seed=seed)
r=s.get_object_from_name('robot'); h=s.get_object_from_name('hook')
print('seed/count/max',seed,info,env.max_steps)
print('robot props',{f:g(s,r,f) for f in ['x','y','theta','base_radius','arm_joint','arm_length','gripper_base_width','gripper_base_height','finger_gap','finger_height','finger_width']})
print('hook props',{f:g(s,h,f) for f in ['x','y','theta','width','length_side1','length_side2','held']})

# aim gripper center at hook center, with initial open then close
for i in range(400):
    rx,ry,th,L = g(s,r,'x'),g(s,r,'y'),g(s,r,'theta'),g(s,r,'arm_length')
    hx,hy=g(s,h,'x'),g(s,h,'y')
    # assumed gripper is L ahead of base. First retract and drive base near hook from above.
    target_theta=math.atan2(hy-ry,hx-rx)
    dist=math.hypot(hx-rx,hy-ry)
    desired_L=min(1.0,max(0.4,dist-0.10))
    a=np.zeros(5,np.float32)
    a[0]=np.clip((hx-0.5*math.cos(target_theta)-rx)*.4,-.03,.03)
    a[1]=np.clip((hy-0.5*math.sin(target_theta)-ry)*.4,-.03,.03)
    a[2]=np.clip(wrap(target_theta-th)*.4,-.098,.098)
    a[3]=np.clip((desired_L-L)*.5,-.08,.08)
    # close only after estimated tip near
    tipx=rx+L*math.cos(th); tipy=ry+L*math.sin(th)
    near=math.hypot(tipx-hx,tipy-hy)<.18
    a[4]=-.015 if near else .015
    s,reward,term,trunc,inf=env.step(a)
    if i%10==0 or g(s,h,'held')>.5:
      print(i,'rob',*[round(g(s,r,f),3) for f in ('x','y','theta','arm_length','finger_gap')], 'hook',*[round(g(s,h,f),3) for f in ('x','y','theta','held')], 'tiperr',round(math.hypot(tipx-hx,tipy-hy),3), 'a',a.tolist())
    if g(s,h,'held')>.5:
      # move base right/up to establish transform
      for j in range(10):
       s,_,_,_,_=env.step(np.array([.03,.02,.05,0,-.015],np.float32))
       print('heldmove',j,'rob',*[round(g(s,r,f),3) for f in ('x','y','theta','arm_length','finger_gap')], 'hook',*[round(g(s,h,f),3) for f in ('x','y','theta','held')])
      break
env.close()
