import numpy as np
from probe_vis_lib import *
# 1) at_home: radius + theta tolerance
env,obs,info=new_env(3)
h=np.array([1.0,-1.75])
lo,hi=0.0,0.5
for _ in range(8):
    m=(lo+hi)/2
    obs,_=goto(env,obs,h[0]+m/np.sqrt(2),h[1]+m/np.sqrt(2),i=0,tol=0.001)
    if rf(obs,0)['at_home']>0.5: lo=m
    else: hi=m
print("1) at_home radius (diagonal) in [%.4f,%.4f]"%(lo,hi))
obs,_=goto(env,obs,h[0],h[1],i=0,tol=0.001)
lo,hi=0.0,0.8
for _ in range(8):
    m=(lo+hi)/2
    # set theta = pi + m
    for _ in range(20):
        d=(np.pi+m-pose(obs,0)[2]+np.pi)%(2*np.pi)-np.pi
        if abs(d)<0.002: break
        obs,_,_,_,_=st(env,dth=float(np.clip(d,-0.4,0.4)),i=0)
    if rf(obs,0)['at_home']>0.5: lo=m
    else: hi=m
print("   at_home theta tolerance in [%.4f,%.4f] rad"%(lo,hi))
env.close()
# 2) image target choice with 3 objectives
env,obs,info=new_env(3)
L=layout(obs); free,xs,res=build_grid(obs)
objs={n:np.array([L[n]['x'],L[n]['y']]) for n in L if n.startswith('objective')}
print("2) objectives",{n:np.round(v,2) for n,v in objs.items()})
# position east: near objective0/1
P=np.array([1.1,0.8])
obs,ok=nav(env,obs,P[0],P[1],i=0,tol=0.03,free=free,xs=xs,res=res)
p=pose(obs,0)[:2]; print("   rover0 at",np.round(p,2),{n:round(float(np.linalg.norm(p-v)),2) for n,v in objs.items()})
for k in range(3):
    obs,_,_,_,_=st(env,op='calibrate',i=0)
    c=rf(obs,0)['calibrated']
    obs,_,_,_,_=st(env,op='image',i=0)
    print("   cycle%d calib=%.0f have=%s"%(k,c,{n:int(feats(obs,n)['have_image_rover0']) for n in objs}))
env.close()
# 3) rover0 sends across the wall (lander at (-1.9,-2)) ; sample an east stone first
env,obs,info=new_env(3)
L=layout(obs); free,xs,res=build_grid(obs); lan=np.array([L['lander']['x'],L['lander']['y']])
east=[n for n in L if n.startswith('sample') and L[n]['x']>0.45 and abs(L[n]['y'])<1.9]
print("3) east samples",east)
if east:
    s=east[0]; sp=np.array([L[s]['x'],L[s]['y']])
    for ang in np.arange(0,2*np.pi,np.pi/4):
        Q=sp+0.18*np.array([np.cos(ang),np.sin(ang)])
        if abs(Q[0])<0.32 or abs(Q[0])>2.05 or abs(Q[1])>2.05: continue
        obs,ok=nav(env,obs,Q[0],Q[1],i=0,tol=0.03,free=free,xs=xs,res=res)
        if np.linalg.norm(pose(obs,0)[:2]-sp)<0.249: break
    obs,_,_,_,_=st(env,op='sample',i=0)
    print("   sampled %s analyzed=%.0f"%(s,feats(obs,s)['analyzed_rover0']))
    for P in [(0.35,-2.0),(0.5,-1.0),(1.5,-1.5)]:
        obs,ok=nav(env,obs,P[0],P[1],i=0,tol=0.04,free=free,xs=xs,res=res)
        p=pose(obs,0)[:2]; d=np.linalg.norm(p-lan)
        obs,_,_,_,_=st(env,op='send',i=0)
        print("   send from %s (lander dist %.2f, wall between) received=%.0f"%(np.round(p,2),d,feats(obs,s)['received_analysis']))
        if feats(obs,s)['received_analysis']>0.5: break
env.close()
