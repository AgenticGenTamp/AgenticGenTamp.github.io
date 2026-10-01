import numpy as np, sys
from probe_lib import *
env=make_env(); obs,info=env.reset(seed=0)
base=rb(obs); p=ppos(obs,'part0')
print("part0",p,"base",base)
obs=grip(env,obs,1.0)
res=[]
for z in [0.35,0.30,0.28,0.26,0.24,0.22,0.20]:
    for dx in [-0.10,-0.05,0.0,0.05,0.10]:
        for dy in [-0.10,-0.05,0.0,0.05,0.10]:
            # retreat up first to avoid path collisions
            obs,_,_=goto(env,obs,(p[0]+dx,p[1]+dy,0.38),Rdown,base,maxsteps=40)
            obs,blk,msg=goto(env,obs,(p[0]+dx,p[1]+dy,z),Rdown,base,maxsteps=40)
            f=fkpos(obs)
            obs=grip(env,obs,-1.0)
            ga=rfeat(obs,'grasp_active')
            if ga>0.5:
                print("GRASP z=%.2f dx=%.2f dy=%.2f fk=%s"%(z,dx,dy,np.round(f,3)),flush=True)
                print("grasp_tf",[round(rfeat(obs,'grasp_tf_'+k),4) for k in ['x','y','z','qx','qy','qz','qw']])
                sys.exit(0)
            obs=grip(env,obs,1.0)
            res.append((z,dx,dy,msg,round(float(f[2]),3)))
    print("z",z,"reach:",[(r[1],r[2],r[3],r[4]) for r in res if r[0]==z],flush=True)
print("no grasp found")
env.close()
