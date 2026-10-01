import numpy as np, sys
exec(open('probe20.py').read().split('env=make_env()')[0])
seed=int(sys.argv[1]); cname=sys.argv[2]; yaw=float(sys.argv[3])
env=make_env(); out=[]
for delta in np.arange(0.165,0.1901,0.0025):
    obs,_=env.reset(seed=seed); q=getq(obs)
    c=obs.get_object_from_name(cname); r=obs.get_object_from_name('robot')
    cp=np.array([obs.get(c,'pose_x')-ARM_OFFSET_X,obs.get(c,'pose_y'),obs.get(c,'pose_z')-H])
    R=down_R(yaw)
    ok=True
    for back in [0.12,0.05,0.0]:
        fl=cp-R@np.array([0,0,delta+back])
        qt,e=ik(q,fl,R); obs,ok=step_to(env,obs,qt); q=getq(obs)
        if not ok: break
    a=np.zeros(11,dtype=np.float32); a[10]=-1; obs,*_=env.step(a)
    out.append((round(delta,4),'B' if not ok else ('G' if obs.get(r,'grasp_active')>0 else '.')))
print(seed,cname,yaw,out)
env.close()
