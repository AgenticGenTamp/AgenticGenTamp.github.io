import numpy as np, fk
from env_client import make_env
from lib_util import robot, step_to
env=make_env()
# map base collision boundary vs near table (centre 4.5, hx .3, hy .6)
for y in [0.0,0.4,0.7,0.9,1.1]:
    obs,_=env.reset(seed=1)
    q=robot(obs)[3:10]
    obs,_,_=step_to(env,obs,[2.5,y,0.0],q)
    # drive +x in 0.05 steps until rejected
    lastx=robot(obs)[0]
    for k in range(60):
        a=np.zeros(11,dtype=np.float32); a[0]=0.05
        prev=robot(obs).copy(); obs,_,_,_,_=env.step(a)
        if abs(robot(obs)[0]-prev[0])<1e-6: break
    print(f"y={y}: max base_x={robot(obs)[0]:.3f}")
# also approach from +x side
for y in [0.0,0.7]:
    obs,_=env.reset(seed=1)
    q=robot(obs)[3:10]
    obs,_,_=step_to(env,obs,[2.5,1.6,0.0],q)
    obs,_,_=step_to(env,obs,[6.0,1.6,0.0],q)
    obs,_,_=step_to(env,obs,[6.0,y,0.0],q)
    for k in range(60):
        a=np.zeros(11,dtype=np.float32); a[0]=-0.05
        prev=robot(obs).copy(); obs,_,_,_,_=env.step(a)
        if abs(robot(obs)[0]-prev[0])<1e-6: break
    print(f"from +x, y={y}: min base_x={robot(obs)[0]:.3f} (base at {robot(obs)[:2]})")
# from -y side driving +y at x=4.5
for x in [4.5,4.2]:
    obs,_=env.reset(seed=1)
    q=robot(obs)[3:10]
    obs,_,_=step_to(env,obs,[2.5,-1.8,0.0],q)
    obs,_,_=step_to(env,obs,[x,-1.8,0.0],q)
    for k in range(60):
        a=np.zeros(11,dtype=np.float32); a[1]=0.05
        prev=robot(obs).copy(); obs,_,_,_,_=env.step(a)
        if abs(robot(obs)[1]-prev[1])<1e-6: break
    print(f"from -y, x={x}: max base_y={robot(obs)[1]:.3f}")
# does base rotation change footprint? rotate 45deg then drive +x at y=0
obs,_=env.reset(seed=1); q=robot(obs)[3:10]
obs,_,_=step_to(env,obs,[2.5,0.0,0.785],q)
for k in range(60):
    a=np.zeros(11,dtype=np.float32); a[0]=0.05
    prev=robot(obs).copy(); obs,_,_,_,_=env.step(a)
    if abs(robot(obs)[0]-prev[0])<1e-6: break
print("rot45 y=0 max base_x", round(float(robot(obs)[0]),3))
env.close()
