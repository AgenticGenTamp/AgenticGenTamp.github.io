import numpy as np
from env_client import make_env
from farroute import robot_state
env = make_env()
for arm in ("home","extended"):
    obs,info = env.reset(seed=3)
    q0 = robot_state(obs)["q"]
    if arm=="extended":
        from approach import ik_solutions, grasp_R
        base=np.array([-3.0,0.0,np.pi])
        sols=ik_solutions(np.array([-3.80,0.0,0.831]),grasp_R(np.pi),base,q0,seeds=10,iters=140)
        qt=sols[0]
        for _ in range(20):
            r=robot_state(obs); dq=np.clip(qt-r["q"],-0.2,0.2)
            if np.max(np.abs(dq))<1e-3: break
            a=np.zeros(11,dtype=np.float32); a[3:10]=dq; obs,*_=env.step(a)
    # rotate to yaw pi and drive west at y=0
    for _ in range(80):
        r=robot_state(obs); a=np.zeros(11,dtype=np.float32)
        a[0]=-0.2; a[1]=np.clip(0.0-r["base"][1],-0.2,0.2)
        a[2]=np.clip((np.pi-r["base"][2]+np.pi)%(2*np.pi)-np.pi,-0.2,0.2)
        obs,*_=env.step(a)
        r2=robot_state(obs)
        if abs(r2["base"][0]-r["base"][0])<1e-6 and r["base"][0]<0: break
    print(arm,"stopped base x=",np.round(robot_state(obs)["base"],3))
env.close()
