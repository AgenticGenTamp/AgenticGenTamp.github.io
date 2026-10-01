import numpy as np
from env_client import make_env


def g(s, name, f):
    return float(s.get(s.get_object_from_name(name), f))


def move(env, s, dx, dy, close=0.0, joints=None):
    a = np.zeros(11, np.float32)
    a[0] = np.clip(dx, -.2, .2)
    a[1] = np.clip(dy, -.2, .2)
    a[10] = close
    if joints is not None:
        a[3:10] = joints
    return env.step(a)


env = make_env(); s, info = env.reset(seed=1)
tx, ty = g(s, "cube0", "pose_x"), g(s, "cube0", "pose_y")
print("target", tx, ty)
for i in range(20):
    bx, by = g(s,"robot","pos_base_x"), g(s,"robot","pos_base_y")
    s,r,t,tr,inf=move(env,s,tx-bx,ty-by)
    print(i, "base",g(s,"robot","pos_base_x"),g(s,"robot","pos_base_y"),"cube",g(s,"cube0","pose_x"),g(s,"cube0","pose_y"),"grasp",g(s,"robot","grasp_active"))
    if abs(g(s,"robot","pos_base_x")-tx)<.01 and abs(g(s,"robot","pos_base_y")-ty)<.01: break
for i in range(3):
    s,r,t,tr,inf=move(env,s,0,0,-1)
    print("close",i,"finger",g(s,"robot","finger_state"),"rg",g(s,"robot","grasp_active"),"cg",g(s,"cube0","grasp_active"),"tf",[g(s,"robot",f"grasp_tf_{q}") for q in "xyz"])
env.close()
