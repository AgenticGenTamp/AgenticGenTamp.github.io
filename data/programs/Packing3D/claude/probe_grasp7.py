import numpy as np
from probe_lib import *
env=make_env(); obs,info=env.reset(seed=0)
base=rb(obs); p=ppos(obs,'part0')
# A: approach grasp pose WITHOUT any close command
obs,_,_=goto(env,obs,(p[0]-0.10,p[1],0.35),Rdown,base)
obs,_,m=goto(env,obs,(p[0]-0.10,p[1],0.30),Rdown,base)
print("A: at grasp pose, no close cmd: grasp=",rfeat(obs,'grasp_active'),"finger",rfeat(obs,'finger_state'))
obs,r,t,tr,i=env.step(np.zeros(11))
print("A2: idle step grasp=",rfeat(obs,'grasp_active'))
# B: send open first
obs=grip(env,obs,1.0,n=2); print("B: after open at grasp pose grasp=",rfeat(obs,'grasp_active'),"finger",rfeat(obs,'finger_state'))
# C: close
obs=grip(env,obs,-1.0,n=1); print("C: after close grasp=",rfeat(obs,'grasp_active'),"finger",rfeat(obs,'finger_state'))
# D: various open magnitudes / many steps
for v in [0.51,0.6,1.0]:
    obs=grip(env,obs,v,n=1)
    print("D open %.2f -> grasp %.1f finger %.3f"%(v,rfeat(obs,'grasp_active'),rfeat(obs,'finger_state')))
for k in range(10):
    obs=grip(env,obs,1.0,n=1)
print("D2 after 10 opens grasp",rfeat(obs,'grasp_active'),"finger",rfeat(obs,'finger_state'))
# E: open together with joint motion
a=np.zeros(11); a[3]=0.05; a[10]=1.0
obs,r,t,tr,i=env.step(a)
print("E open+motion grasp",rfeat(obs,'grasp_active'),"finger",rfeat(obs,'finger_state'))
# F: lift, move over rack, place on rack, open
obs,_,_=goto(env,obs,(p[0]-0.10,p[1],0.40),Rdown,base)
print("F lifted part",np.round(ppos(obs,'part0'),3),"grasp",rfeat(obs,'grasp_active'))
obs,_,m=goto(env,obs,(0.30+0.10,0.0,0.40),Rdown,base)   # part center over rack center
print("F over rack",m,"part",np.round(ppos(obs,'part0'),3))
for z in [0.38,0.36,0.35,0.34,0.33,0.32,0.31]:
    obs,blk,m=goto(env,obs,(0.40,0.0,z),Rdown,base)
    print("  z",z,m,"part",np.round(ppos(obs,'part0'),3))
    if blk: break
    obs2=grip(env,obs,1.0,n=1)
    if rfeat(obs2,'grasp_active')<0.5:
        print("  RELEASED at z",z,"part",np.round(ppos(obs2,'part0'),3)); obs=obs2; break
    obs=obs2
print("F end grasp",rfeat(obs,'grasp_active'),"part",np.round(ppos(obs,'part0'),3))
env.close()
