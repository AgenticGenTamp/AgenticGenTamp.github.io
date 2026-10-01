import sys, numpy as np
from env_client import make_env
from farroute import FarRoute, robot_state, _cart_path
from approach import world_fk, grasp_R, dq_wrap

seed = int(sys.argv[1]) if len(sys.argv)>1 else 1
env = make_env(); obs, info = env.reset(seed=seed)
pol = FarRoute(); pl = pol.reset(obs, info)
print("pick", pl["name"], "blk", np.round(pl["block"],3), "base", np.round(pl["base"],3))
for t in range(60):
    a = pol.get_action(obs)
    obs, rew, term, trunc, info = env.step(a)
    if pol.phase=="approach" and pol.stall>=2: break
r = robot_state(obs)
p,R = world_fk(r["q"], r["base"])
print("stalled at step",t,"tool",np.round(p,3),"base",np.round(r["base"],3))
print("target grasp", np.round(pl["path"][-1] is not None and world_fk(pl["path"][-1], pl["base"])[0],3))
print("wp remaining", len(pol.wp))
# try single cartesian nudges
for name,d in [("+x",[.03,0,0]),("-x",[-.03,0,0]),("+y",[0,.03,0]),("-y",[0,-.03,0]),("+z",[0,0,.03]),("-z",[0,0,-.03])]:
    q0 = robot_state(obs)["q"]; b0 = robot_state(obs)["base"]
    pth = _cart_path(b0, q0, [world_fk(q0,b0)[0]+np.array(d)], R)
    if not pth: print(" ",name,"IK fail"); continue
    act = np.zeros(11,dtype=np.float32); act[3:10]=np.clip(dq_wrap(pth[0]-q0),-0.2,0.2)
    o2,_,_,_,_ = env.step(act)
    moved = not np.allclose(robot_state(o2)["q"], q0, atol=1e-6)
    print(" ",name,"moved",moved, np.round(world_fk(robot_state(o2)["q"],b0)[0],3))
    if moved:
        act[3:10] *= -1
        for _ in range(3):
            o2,_,_,_,_ = env.step(np.zeros(11,dtype=np.float32)*0+act)
        obs = o2
        print("    back to", np.round(world_fk(robot_state(obs)["q"],b0)[0],3))
    else:
        obs = o2
env.close()
