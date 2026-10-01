from env_client import make_env
import numpy as np, math
env=make_env()

def goto(env,obs,tx,ty,th,steps=200):
    r=obs.get_object_from_name("robot")
    for _ in range(steps):
        x,y,t=[float(obs.get(r,f)) for f in ["x","y","theta"]]
        d=(th-t+math.pi)%(2*math.pi)-math.pi
        dx=np.clip(tx-x,-0.05,0.05); dy=np.clip(ty-y,-0.05,0.05)
        if abs(dx)<1e-6 and abs(dy)<1e-6 and abs(d)<1e-4: break
        obs,_,_,_,_=env.step(np.array([dx,dy,np.clip(d,-0.19,0.19),-0.1,0.0]))
    return obs

# seed 5: wall x=0.5, lower seg [0,1.058], upper seg [1.315,2.5]
obs,info=env.reset(seed=5)
r=obs.get_object_from_name("robot")
# approach from left at various y, find max x
for y in [1.1865, 1.30, 1.31, 1.35, 1.4, 1.5]:
    obs,_=env.reset(seed=5)
    obs=goto(env,obs,0.30,y,0.0)
    prev=(float(obs.get(r,"x")),float(obs.get(r,"y")))
    for i in range(300):
        obs,_,_,_,_=env.step(np.array([0.001,0,0,-0.1,0.0]))
        cur=(float(obs.get(r,"x")),float(obs.get(r,"y")))
        if abs(cur[0]-prev[0])<1e-7: break
        prev=cur
    print("theta=0  y=%.4f maxx=%.4f clearance=%.4f"%(prev[1],prev[0],0.5-prev[0]))
for y in [1.1865, 1.35]:
    obs,_=env.reset(seed=5)
    obs=goto(env,obs,0.30,y,math.pi)
    prev=(float(obs.get(r,"x")),float(obs.get(r,"y")))
    for i in range(300):
        obs,_,_,_,_=env.step(np.array([0.001,0,0,-0.1,0.0]))
        cur=(float(obs.get(r,"x")),float(obs.get(r,"y")))
        if abs(cur[0]-prev[0])<1e-7: break
        prev=cur
    print("theta=pi y=%.4f maxx=%.4f clearance=%.4f"%(prev[1],prev[0],0.5-prev[0]))
env.close()
