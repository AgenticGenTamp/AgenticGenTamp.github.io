import numpy as np
from probe_lib import *
env=make_env(); obs,info=env.reset(seed=0)
base=rb(obs); p=ppos(obs,'part0')
def trial(dx,dy,z,R):
    global obs
    obs,_,_=goto(env,obs,(p[0]+dx,p[1]+dy,0.40),R,base,maxsteps=40)
    obs,blk,m=goto(env,obs,(p[0]+dx,p[1]+dy,z),R,base,maxsteps=40)
    f=fkpos(obs)
    obs=grip(env,obs,-1.0,n=1)
    ga=rfeat(obs,'grasp_active')
    if ga>0.5:
        # release is impossible off-rack; must reset env
        return True,f
    return False,f
def scan(vals,mk,label,R):
    out=[]
    for v in vals:
        dx,dy,z=mk(v)
        g,f=trial(dx,dy,z,R)
        out.append((round(v,3),int(g),round(float(f[2]),3)))
        if g:
            print(label,"GRASP at",round(v,3),"fk",np.round(f,3),flush=True)
            return out,v
    print(label,out,flush=True)
    return out,None
print("part",p)
# z scan at dx=-0.10 dy=0 : find z range (need fresh env each grasp -> instead find first grasp from top)
zs=np.arange(0.36,0.23,-0.01)
found=[]
for z in zs:
    g,f=trial(-0.10,0.0,z,Rdown)
    found.append((round(z,3),int(g)))
    if g: break
print("z scan (dx=-0.10,dy=0):",found,flush=True)
env.close()
# now with fresh env: find z upper/lower by scanning from bottom
env=make_env(); obs,info=env.reset(seed=0); base=rb(obs); p=ppos(obs,'part0')
found=[]
for z in np.arange(0.24,0.37,0.01):
    g,f=trial(-0.10,0.0,z,Rdown)
    found.append((round(z,3),int(g)))
    if g: break
print("z scan up:",found,flush=True)
env.close()
# dx scan at z=0.30
env=make_env(); obs,info=env.reset(seed=0); base=rb(obs); p=ppos(obs,'part0')
found=[]
for dx in np.arange(-0.20,0.06,0.02):
    g,f=trial(dx,0.0,0.30,Rdown)
    found.append((round(dx,3),int(g)))
    if g: break
print("dx scan z0.30:",found,flush=True)
env.close()
env=make_env(); obs,info=env.reset(seed=0); base=rb(obs); p=ppos(obs,'part0')
found=[]
for dy in np.arange(-0.10,0.11,0.02):
    g,f=trial(-0.10,dy,0.30,Rdown)
    found.append((round(dy,3),int(g)))
    if g: break
print("dy scan:",found,flush=True)
env.close()
# yaw 90
env=make_env(); obs,info=env.reset(seed=0); base=rb(obs); p=ppos(obs,'part0')
R=Rdown@rotz(np.pi/2)
found=[]
for dy in np.arange(-0.16,0.17,0.02):
    g,f=trial(0.0,dy,0.30,R)
    found.append((round(dy,3),int(g)))
    if g: break
print("yaw90 dy scan (dx=0,z0.30):",found,flush=True)
env.close()
