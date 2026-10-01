import numpy as np
from helper import Sim, wrap, R
np.set_printoptions(precision=4,suppress=True)
sim=Sim(seed=4)
sim.grasp_hook(d=0.60)
o=sim.obs
thr=o[2]; rel=R(-thr)@(o[9:11]-o[:2]); dth=wrap(o[11]-thr)
M=o[20:22].copy()
def hp(o):
    C=o[9:11].copy(); t=o[11]
    a=np.array([-np.cos(t),-np.sin(t)]); b=np.array([-a[1],a[0]])
    return C,a,b
t_goal=np.pi/2
sim.goto(o[0],o[1],wrap(t_goal-dth),0.2,1.0,maxn=60)
o=sim.obs; C,a,b=hp(o); print("after rot C",C,"a",a,"b",b,"robot",o[:3])
q=0.30; s_off=-0.30
Cd = M - q*b - s_off*a
pos_goal = Cd - R(o[2])@rel
print("target C",Cd,"robot goal",pos_goal)
def move_axis(sim,tx,ty):
    for ax in (0,1):
        for k in range(200):
            o=sim.obs
            d=(tx-o[0]) if ax==0 else (ty-o[1])
            if abs(d)<1e-4: break
            st=np.clip(d,-0.05,0.05)
            act=[st if ax==0 else 0, st if ax==1 else 0,0,0,1.0]
            p=o[:2].copy(); sim.step(act)
            if np.abs(sim.obs[:2]-p).max()<1e-9: print("  blocked ax",ax,sim.obs[:2]); break
move_axis(sim,pos_goal[0],pos_goal[1])
o=sim.obs; C,a,b=hp(o)
print("now robot",o[:3],"C",C,"projA",np.dot(o[20:22]-C,a),"projB",np.dot(o[20:22]-C,b))
prevM=o[20:22].copy()
for i in range(80):
    sim.step([0,0.005,0,0,1.0])
    o=sim.obs; C,a,b=hp(o)
    pa=np.dot(o[20:22]-C,a); pb=np.dot(o[20:22]-C,b)
    moved=np.linalg.norm(o[20:22]-prevM)
    if moved>1e-7 or i%10==0:
        print(f"i={i} ry={o[1]:.4f} projA={pa:.4f} projB={pb:.4f} dM={o[20:22]-prevM}")
    prevM=o[20:22].copy()
    if sim.term: print("TERM"); break
sim.close()
