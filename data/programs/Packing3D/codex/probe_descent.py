from env_client import make_env
import numpy as np

def v(s,n,f): return s.get(s.get_object_from_name(n),f)

def run(which):
    env=make_env(); s,_=env.reset(seed=0,options={"object_count":1})
    dx=v(s,'part0','pose_x')-v(s,'rack','pose_x')
    dy=v(s,'part0','pose_y')-v(s,'rack','pose_y')
    for _ in range(2):
        a=np.zeros(11,np.float32);a[0]=np.clip(dx,-.2,.2);a[1]=np.clip(dy,-.2,.2);a[10]=1
        s,*_=env.step(a);dx-=a[0];dy-=a[1]
    for i in range(10):
        a=np.zeros(11,np.float32);a[10]=-1
        if which=='j4': a[6]=-.1
        elif which=='j2': a[4]=.1
        else: a[4]=.06;a[6]=-.1
        s,*_=env.step(a)
        print(which,i,"j2",round(v(s,'robot','joint_2'),2),"j4",round(v(s,'robot','joint_4'),2),
              "g",v(s,'robot','grasp_active'),"p",[round(v(s,'part0',f),3) for f in ('pose_x','pose_y','pose_z')])
        if v(s,'robot','grasp_active'): break
    env.close()
for w in ('j4','j2','both'):run(w)
