import numpy as np, sys
import kinova
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)

MOUNT = np.array([0.0,0.0,0.45])
TOOL = 0.13
Rdown = np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)

def world_to_arm(p, base):
    yaw = base[2]
    c,s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return R.T @ (p - np.array([base[0],base[1],0.0])) - MOUNT

def run(zpred, sweep=True):
    env = make_env()
    obs,_ = env.reset(seed=0)
    b0 = obs[54:57].copy()
    base = obs[16:19].copy()
    traj=[]
    # go to start point
    for phase,pt in enumerate([np.array([b0[0]-0.15,b0[1],0.25]),
                               np.array([b0[0]-0.15,b0[1],zpred]),
                               np.array([b0[0]+0.15,b0[1],zpred])]):
        n = 80 if phase<2 else 120
        for i in range(n):
            q = obs[19:26]
            pa = world_to_arm(pt, obs[16:19])
            dq = kinova.ik_step(q, pa, Rdown, tool_offset=TOOL)
            a = np.zeros(11); a[3:10]=np.clip(dq,-0.1,0.1); a[10]=0.0
            obs,r,te,tr,_ = env.step(a.astype(np.float32))
        T = kinova.fk(obs[19:26], TOOL)
        traj.append((phase, T[:3,3].copy(), obs[54:57].copy()))
    env.close()
    return traj, b0

if __name__=="__main__":
    z = float(sys.argv[1]) if len(sys.argv)>1 else 0.05
    traj,b0 = run(z)
    print("zpred",z,"block0",b0)
    for p,ee,b in traj:
        print(" phase",p,"ee_arm",ee,"block",b)
