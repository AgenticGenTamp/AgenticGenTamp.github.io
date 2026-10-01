"""Measure folded-wrist clearance posture for shallow-east failures."""
import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

def g(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def vec(s,prefix): return np.array([g(s,"robot",prefix+x) for x in "xyz"])
def Rquat(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def act(e,s,d):
    a=np.zeros(11,np.float32);a[10]=1
    for i,v in d.items():a[i]=np.clip(v,-.2,.2)
    return e.step(a)[0]
def movej(e,s,p,idx,goal):
    f='joint_'+str(idx-2)
    for _ in range(30):
        d=goal-g(s,'robot',f)
        if idx in (7,9):d=p.w(d)
        if abs(d)<1e-4:break
        s=act(e,s,{idx:d})
    return s
def setup(seed):
    e=make_env();s,info=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
    for _ in range(100):
        if p.stage==5:break
        s,*_=e.step(p.get_action(s))
    # execute only lifted base positioning
    for _ in range(30):
        a=p.get_action(s)
        if a[4]>.01:break
        s,*_=e.step(a)
    s=movej(e,s,p,7,p.w(p.Q[4]+1.4))
    s=movej(e,s,p,9,p.w(p.Q[6]+.4))
    s=movej(e,s,p,4,p.Q[1])
    return e,s,p
def measure(seed):
    e,s,p=setup(seed)
    q=np.array([g(s,'robot','grasp_tf_q'+x) for x in 'xyzw']);R=Rquat(q)
    tool=vec(s,'grasp_tf_'); block=np.array([g(s,'green0','pose_'+x) for x in 'xyz'])
    print(seed,'base',p.robot(s),'yaw',g(s,'robot','base_rot'))
    print('q',[round(g(s,'robot','joint_'+str(j)),5) for j in range(1,8)])
    print('tool',tool,'block',block,'delta',block-tool,'local',R.T@(block-tool),'R',R,sep='\n')
    # Directly translate base by horizontal block-tool delta, keeping folded posture.
    goal=p.robot(s)+(block-tool)[:2]
    for _ in range(15):
        d=goal-p.robot(s)
        if max(abs(d))<1e-4:break
        old=p.robot(s).copy();s=act(e,s,{0:d[0],1:d[1]})
        if max(abs(p.robot(s)-old))<1e-7:print('translation rejected');break
    q=np.array([g(s,'robot','grasp_tf_q'+x) for x in 'xyzw']);R=Rquat(q);tool=vec(s,'grasp_tf_')
    print('after base',p.robot(s),'tool',tool,'local',R.T@(block-tool))
    a=np.zeros(11,np.float32);a[10]=-1;s,*_=e.step(a)
    print('HELD',g(s,'green0','grasp_active'))
    e.close()
if __name__=='__main__':
    for x in sys.argv[1:] or ['101','191']:measure(int(x))
