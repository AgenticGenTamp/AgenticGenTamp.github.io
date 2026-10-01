import numpy as np, sys
exec(open('probe20.py').read().split('env=make_env()')[0])
delta=float(sys.argv[1]); axis=sys.argv[2]
env=make_env(); out=[]
for off in np.arange(-0.06,0.061,0.01):
    obs,_=env.reset(seed=0); q=getq(obs)
    c=obs.get_object_from_name('cube0'); r=obs.get_object_from_name('robot')
    cp=np.array([obs.get(c,'pose_x')-ARM_OFFSET_X,obs.get(c,'pose_y'),obs.get(c,'pose_z')-H])
    R=down_R(np.pi/2)
    lo=np.array([0,0,delta]); lo[0 if axis=='x' else 1]=off
    ok=True
    for back in [0.12,0.05,0.0]:
        fl=cp-R@(lo+np.array([0,0,back]))
        qt,e=ik(q,fl,R); obs,ok=step_to(env,obs,qt); q=getq(obs)
        if not ok: break
    a=np.zeros(11,dtype=np.float32); a[10]=-1; obs,*_=env.step(a)
    out.append('B' if not ok else ('G' if obs.get(r,'grasp_active')>0 else '.'))
print(axis,delta,''.join(out))
env.close()
