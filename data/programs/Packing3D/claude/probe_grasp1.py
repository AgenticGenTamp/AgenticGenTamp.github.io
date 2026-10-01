import numpy as np
from probe_lib import *
env=make_env(); obs,info=env.reset(seed=0)
base=rb(obs)
names=sorted([n for n in obs.get_object_names() if n.startswith('part')])
print("parts",names,"base",base)
for n in names:
    o=obs.get_object_from_name(n)
    print(n,o.type.name,[round(obs.get(o,f),4) for f in obs.type_features[o.type]])
r=obs.get_object_from_name('robot')
print("robot feats",obs.type_features[r.type])
print("robot vals",[round(obs.get(r,f),4) for f in obs.type_features[r.type]])
p=ppos(obs,names[0]); print("target part",names[0],p)
print("fk at start",fkpos(obs))
# open gripper first
obs=grip(env,obs,1.0)
print("finger after open",rfeat(obs,'finger_state'))
obs=grip(env,obs,-1.0)
print("finger after close(free space)",rfeat(obs,'finger_state'),"grasp",rfeat(obs,'grasp_active'))
obs=grip(env,obs,1.0)
for z in np.arange(0.40,0.17,-0.02):
    obs,blk,msg=goto(env,obs,(p[0],p[1],z),Rdown,base)
    if blk:
        print("z=%.3f %s fk=%s"%(z,msg,np.round(fkpos(obs),3))); continue
    obs=grip(env,obs,-1.0)
    ga=rfeat(obs,'grasp_active'); fs=rfeat(obs,'finger_state'); pga=pfeat(obs,names[0],'grasp_active')
    print("z=%.3f ok grasp=%.2f partga=%.2f finger=%.3f"%(z,ga,pga,fs))
    if ga>0.5:
        print("GRASP! fk",fkpos(obs),"grasp_tf",[round(rfeat(obs,'grasp_tf_'+k),4) for k in ['x','y','z','qx','qy','qz','qw']])
        break
    obs=grip(env,obs,1.0)
env.close()
