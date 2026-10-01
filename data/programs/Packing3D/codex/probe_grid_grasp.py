from env_client import make_env
import numpy as np

def gv(s,n,f): return s.get(s.get_object_from_name(n),f)
for q2 in [-.15,.05,.25,.45,.65,.85]:
    env=make_env();s,_=env.reset(seed=0,options={"object_count":1})
    alignx=-.12+gv(s,'part0','pose_x')-gv(s,'rack','pose_x'); aligny=gv(s,'part0','pose_y')
    # Set shoulder while opening and move roughly into the region.
    while abs(gv(s,'robot','joint_2')-q2)>.01:
        a=np.zeros(11,np.float32);a[4]=np.clip(q2-gv(s,'robot','joint_2'),-.2,.2);a[10]=1
        s,*_=env.step(a)
    xs=np.linspace(max(-.42,alignx-.20),min(-.12,alignx+.12),9)
    ys=np.linspace(aligny-.20,aligny+.20,9)
    for ix,x in enumerate(xs):
        yseq=ys if ix%2==0 else ys[::-1]
        for y in yseq:
            # arrive exactly (large moves are limited) then close
            for _ in range(2):
                a=np.zeros(11,np.float32);a[0]=np.clip(x-gv(s,'robot','pos_base_x'),-.2,.2);a[1]=np.clip(y-gv(s,'robot','pos_base_y'),-.2,.2);a[10]=-1
                s,*_=env.step(a)
            if gv(s,'robot','grasp_active'):
                print('SUCCESS q2',q2,'base',gv(s,'robot','pos_base_x'),gv(s,'robot','pos_base_y'),'cell',x,y,'tf',[gv(s,'robot','grasp_tf_'+c) for c in 'xyz'])
                break
        if gv(s,'robot','grasp_active'): break
    print('done',q2,'g',gv(s,'robot','grasp_active'),flush=True);env.close()
    if gv(s,'robot','grasp_active'): break
