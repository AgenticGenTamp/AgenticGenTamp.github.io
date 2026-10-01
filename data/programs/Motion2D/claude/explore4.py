from env_client import make_env
import numpy as np, math
env = make_env()

def probe(seed, theta_target, arm_action, direction, name):
    obs,info=env.reset(seed=seed)
    r=obs.get_object_from_name("robot")
    # set theta
    for _ in range(40):
        th=float(obs.get(r,"theta"))
        d=(theta_target-th+math.pi)%(2*math.pi)-math.pi
        if abs(d)<1e-3: break
        obs,_,_,_,_=env.step(np.array([0,0,np.clip(d,-0.196,0.196),arm_action,0.0]))
    for _ in range(10):
        obs,_,_,_,_=env.step(np.array([0,0,0,arm_action,0.0]))
    # push
    prev=None
    for i in range(200):
        obs,rew,t,tr,_=env.step(np.array([direction[0]*0.05,direction[1]*0.05,0,arm_action,0.0]))
        cur=(float(obs.get(r,"x")),float(obs.get(r,"y")))
        if prev==cur: break
        prev=cur
    # then fine steps
    for i in range(60):
        obs,rew,t,tr,_=env.step(np.array([direction[0]*0.005,direction[1]*0.005,0,arm_action,0.0]))
        cur=(float(obs.get(r,"x")),float(obs.get(r,"y")))
        if prev==cur: break
        prev=cur
    print(name, "final", round(prev[0],4), round(prev[1],4), "arm_joint",round(float(obs.get(r,"arm_joint")),3),"theta",round(float(obs.get(r,"theta")),3))

# wall at x=0.5 for seed 0
probe(0, 0.0, -0.1, (1,0), "toward wall, arm retracted, facing +x")
probe(0, math.pi, -0.1, (1,0), "toward wall, arm retracted, facing -x")
probe(0, 0.0, 0.1, (1,0), "toward wall, arm extended, facing +x")
probe(0, 0.0, -0.1, (0,1), "up, arm retracted")
probe(0, 0.0, -0.1, (0,-1), "down, arm retracted")
probe(0, 0.0, -0.1, (-1,0), "left, arm retracted")
env.close()
