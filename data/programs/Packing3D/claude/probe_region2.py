import numpy as np
from plib2 import *
from env_client import make_env
env=make_env()
def run(part,delta,R,seed=0):
    obs,info=env.reset(seed=seed); base=rb(obs)
    obs,g,m=try_grasp(env,obs,part,delta,R,base)
    return g,m
print("center check",run('part0',[0,0,0],Rdown),flush=True)
for ax,name in [(0,'x'),(1,'y'),(2,'z')]:
    row=[]
    for v in np.arange(-0.07,0.071,0.01):
        d=[0,0,0]; d[ax]=float(v)
        g,m=run('part0',d,Rdown)
        row.append((round(v,3),int(g)))
    print("axis",name,row,flush=True)
# yaw dependence at center
for yd in [45,90,180]:
    R=Rdown@rotz(np.radians(yd))
    print("yaw",yd,run('part0',[0,0,0],R),flush=True)
# triangle part1
print("triangle center",run('part1',[0,0,0],Rdown),flush=True)
for yd in [0,90]:
    R=Rdown@rotz(np.radians(yd))
    for v in [-0.03,0.0,0.03]:
        print("tri yaw",yd,"dx",v,run('part1',[v,0,0],R),flush=True)
env.close()
