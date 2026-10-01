from th import *
import sys
yaw=float(sys.argv[1]); axis=int(sys.argv[2])
env=make_env(); 
for seed in range(50):
    obs,_=env.reset(seed=seed, options={'object_count':0})
    tb=obs.get_object_from_name('target_block'); tr=obs.get_object_from_name('target_region')
    bx,by=obs.get(tb,'pose_x'),obs.get(tb,'pose_y'); rx,ry=obs.get(tr,'pose_x'),obs.get(tr,'pose_y')
    if 0.22<bx<0.38 and abs(by)<0.25 and np.hypot(bx-rx,by-ry)>0.2: break
h=H(seed, env=env, oc=0); name='target_block'
p=h.pose(name); he=h.he(name); top=p[2]+he[2]
out=[]
for d in np.arange(0.0, 0.10, 0.01):
    xy=p[:2].copy(); xy[axis]+=d
    ok0,_=h.goto([xy[0],xy[1],top+0.08],yaw)
    if not ok0: out.append(f'{d:.2f}:NA'); continue
    z=top+0.08
    while z>top-0.08:
        z-=0.003
        ok,_=h.goto([xy[0],xy[1],z],yaw)
        if not ok: break
    out.append(f'{d:.2f}:{z+0.003-top:+.3f}')
    z=top+0.08
    while h.tool()[2,3] < top+0.075:
        ok,_=h.goto([xy[0],xy[1],h.tool()[2,3]+0.01],yaw)
        if not ok: print('stuck'); break
print('seed',seed,'he',he.round(3),'yaw',yaw,'axis',axis,' '.join(out))
