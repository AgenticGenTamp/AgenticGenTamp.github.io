import numpy as np
from probe_vis_lib import *
# ---------- A) at_home tolerance ----------
env,obs,info=new_env(1)
h0=np.array([1.0,-1.75])
print("A) at_home start:",rf(obs,0)['at_home'],rf(obs,1)['at_home'])
for dx in [0.05,0.10,0.15,0.20,0.25,0.30]:
    obs,_=goto(env,obs,h0[0]+dx,h0[1],i=0,tol=0.002)
    print("   dx=%.2f  pos=%.3f at_home=%.0f"%(dx,pose(obs,0)[0],rf(obs,0)['at_home']))
obs,_=goto(env,obs,h0[0],h0[1],i=0,tol=0.002)
print("   back home at_home=%.0f"%rf(obs,0)['at_home'])
for dy in [0.10,0.20,0.25,0.30]:
    obs,_=goto(env,obs,h0[0],h0[1]+dy,i=0,tol=0.002)
    print("   dy=%.2f at_home=%.0f"%(dy,rf(obs,0)['at_home']))
obs,_=goto(env,obs,h0[0],h0[1],i=0,tol=0.002)
for k in range(4):
    obs,_,_,_,_=st(env,dth=0.4,i=0)
print("   after rotating theta=%.2f at_home=%.0f"%(pose(obs,0)[2],rf(obs,0)['at_home']))
# fine bisect in x
lo,hi=0.0,0.30
for _ in range(7):
    m=(lo+hi)/2
    obs,_=goto(env,obs,h0[0]+m,h0[1],i=0,tol=0.001)
    if rf(obs,0)['at_home']>0.5: lo=m
    else: hi=m
print("   at_home x-tolerance in [%.4f,%.4f]"%(lo,hi))
env.close()
# ---------- B/C) send: lander visibility range, store/drop semantics ----------
env,obs,info=new_env(1)
L=layout(obs); lan=np.array([L['lander']['x'],L['lander']['y']])
free,xs,res=build_grid(obs)
s1=np.array([L['sample1']['x'],L['sample1']['y']])   # stone, west
s0=np.array([L['sample0']['x'],L['sample0']['y']])   # stone, west
print("B) lander",lan,"sample1",s1,"is_soil",L['sample1']['is_soil'])
obs,ok=nav(env,obs,s1[0]+0.15,s1[1]+0.15,i=1,tol=0.02,free=free,xs=xs,res=res)
obs,_,_,_,_=st(env,op='sample',i=1)
print("   sampled: store=%.0f analyzed_r1=%.0f"%(rf(obs,1)['store_full'],feats(obs,'sample1')['analyzed_rover1']))
# go far north-west then approach lander, calling send each step
obs,ok=nav(env,obs,-0.45,2.0,i=1,tol=0.03,free=free,xs=xs,res=res)
p=pose(obs,1)[:2]; print("   far pos",np.round(p,2),"dist to lander %.3f"%np.linalg.norm(p-lan))
u=(lan-p)/np.linalg.norm(lan-p)
d=np.linalg.norm(p-lan); prev=d
for k in range(60):
    obs,_,_,_,_=st(env,op='send',i=1)
    if feats(obs,'sample1')['received_analysis']>0.5:
        print("   SEND first succeeded at lander-dist=%.3f (prev fail %.3f) store=%.0f"%(d,prev,rf(obs,1)['store_full'])); break
    prev=d; d-=0.05
    tgt=lan+ (p-lan)/np.linalg.norm(p-lan)*d
    obs,_=goto(env,obs,tgt[0],tgt[1],i=1,tol=0.01)
    d=np.linalg.norm(pose(obs,1)[:2]-lan)
else: print("   send never succeeded, d=",d)
print("   after send: store_full=%.0f analyzed=%.0f received=%.0f"%(rf(obs,1)['store_full'],feats(obs,'sample1')['analyzed_rover1'],feats(obs,'sample1')['received_analysis']))
# C) drop then send for a second sample
obs,_,_,_,_=st(env,op='drop',i=1); print("C) after drop store=%.0f"%rf(obs,1)['store_full'])
obs,ok=nav(env,obs,s0[0]+0.15,s0[1]+0.1,i=1,tol=0.03,free=free,xs=xs,res=res)
obs,_,_,_,_=st(env,op='sample',i=1)
print("   sample0 analyzed=%.0f store=%.0f"%(feats(obs,'sample0')['analyzed_rover1'],rf(obs,1)['store_full']))
obs,_,_,_,_=st(env,op='drop',i=1)
print("   dropped: store=%.0f analyzed=%.0f"%(rf(obs,1)['store_full'],feats(obs,'sample0')['analyzed_rover1']))
obs,ok=nav(env,obs,lan[0]+0.9,lan[1]+0.9,i=1,tol=0.05,free=free,xs=xs,res=res)
obs,_,_,_,_=st(env,op='send',i=1)
print("   send after drop: received_analysis(sample0)=%.0f (dist %.2f)"%(feats(obs,'sample0')['received_analysis'],np.linalg.norm(pose(obs,1)[:2]-lan)))
# rover0 (other side of wall) send test: it holds nothing -> also test occlusion of lander later
env.close()
