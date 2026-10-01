"""Focused shallow-east green grasp search; never imported by approach."""
import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

def g(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def vals(s,p):
    return np.r_[p.robot(s),g(s,'robot','base_rot'),
                 [g(s,'robot','joint_'+str(i)) for i in range(1,8)]]

def setup(seed):
    e=make_env();s,info=e.reset(seed=seed)
    p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
    for _ in range(70):
        if p.stage==5: break
        s,*_=e.step(p.get_action(s))
    # Run only tangent base translation with lifted arm, stopping when stuck.
    for _ in range(20):
        old=vals(s,p); ns,*_=e.step(p.get_action(s))
        if np.max(np.abs(vals(ns,p)-old))<1e-6: s=ns;break
        s=ns
    return e,s,p

def move(e,s,p,target,close=False,steps=30,rate=.10):
    for _ in range(steps):
        cur=vals(s,p); d=target-cur
        a=np.zeros(11,np.float32);a[:10]=np.clip(d,-rate,rate);a[10]=-1 if close else 1
        ns,*_=e.step(a);s=ns
        if g(s,'green0','grasp_active')>.5:return s,True
        if np.max(np.abs(d))<.002:break
    return s,False

def one(seed,base_delta,qdelta):
    e,s,p=setup(seed); initial=vals(s,p)
    target=initial.copy();target[:2]+=base_delta;target[3:]+=qdelta
    # use open motion to target, but close throughout final small q2 descent
    pre=target.copy();pre[4]=initial[4]
    s,_=move(e,s,p,pre,False,35,.08)
    s,ok=move(e,s,p,target,True,20,.025)
    actual=vals(s,p);tf=np.array([g(s,'robot','grasp_tf_'+x) for x in 'xyz'])
    green=np.array([g(s,'green0','pose_'+x) for x in 'xyz'])
    e.close();return ok,actual,tf-green

if __name__=='__main__':
 seed=int(sys.argv[1]) if len(sys.argv)>1 else 101
 # Deterministic low-dimensional sweep. qdelta is relative to lifted Q posture.
 tests=[]
 for bx in (0,-.03,-.06,-.10):
  for by in np.arange(-.16,.161,.04):
   for q1 in (-.16,-.08,0,.08,.16):
    for q4 in (-.20,-.10,0,.10,.20):
     for lower in (.04,.08,.12,.16,.20):
      q=np.zeros(7);q[0]=q1;q[1]=lower;q[3]=q4
      tests.append((np.array([bx,by]),q))
 for k,(bd,qd) in enumerate(tests):
  ok,a,err=one(seed,bd,qd)
  if ok:
   print('HIT',seed,'k',k,'base_delta',bd.tolist(),'qdelta',qd.tolist(),
         'actual',a.tolist(),'tf_minus_green',err.tolist(),flush=True);break
  if k%100==0:print('progress',k,'actual',np.round(a,3).tolist(),'err',np.round(err,3).tolist(),flush=True)
 else:print('NONE',seed,len(tests))
