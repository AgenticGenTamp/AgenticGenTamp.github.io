import numpy as np, mujoco, sys
sys.path.insert(0,'/sandbox')
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
from approach import GeneratedApproach, _quat_yaw, _wrap
env = KinderGeom3DEnv("kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0")
re = env._kinder_env._object_centric_env._robot_env
class H:
    def __init__(s, seed):
        s.obs,_=env.reset(seed=seed)
        s.ap=GeneratedApproach(env.action_space, env.observation_space, {}); s.ap.reset(s.obs,{})
        s.base=s.obs[93:96].copy(); s.q=s.obs[96:103].copy(); s.grip=0.0
        s.m,s.d=re.sim.model.mj_model,re.sim.data.mj_data
        s.bid=mujoco.mj_name2id(s.m,mujoco.mjtObj.mjOBJ_BODY,"boxed_drink_0")
        s.log=[]
    def drink(s): return s.obs[16:19].copy()
    def pinch(s): return s.ap._pinch(s.obs[96:103], s.obs[93:96])
    def step(s, n=1):
        for _ in range(n):
            a=np.zeros(11); b=s.obs[93:96]; e=s.base-b; e[2]=_wrap(e[2]); a[:3]=np.clip(e,-.1,.1)
            a[3:10]=np.clip(s.q-s.obs[96:103],-.1,.1); a[10]=s.grip
            s.obs,*_=env.step(a.astype(np.float32))
    def setbase(s, base, n=60):
        s.base=np.array(base,float); 
        # keep pinch where it is in world
        p,y=s.ap._pinch(s.q, s.obs[93:96]) 
        for _ in range(n):
            s.step(); 
            if np.abs(s.obs[93:95]-s.base[:2]).max()<0.003: break
    def move(s, p1, yaw, speed=0.01, settle=20):
        p0,y0=s.ap._pinch(s.q, s.base); dy=_wrap(yaw-y0)
        n=max(1,int(np.ceil(max(np.linalg.norm(np.array(p1)-p0)/speed, abs(dy)/0.08))))
        for k in range(1,n+1):
            f=k/n; s.q=s.ap.ik(s.base, p0+f*(np.array(p1)-p0), y0+f*dy, s.q, iters=60); s.step()
        s.step(settle)
    def close(s, n=15): s.grip=1.0; s.step(n)
    def open(s, n=15): s.grip=0.0; s.step(n)
    def rep(s, tag=""):
        p,y=s.pinch(); print(f"{tag:10s} drink {np.round(s.drink(),3)} dyaw {_quat_yaw(s.obs[19:23]):.2f} pinch {np.round(p,3)} yaw {y:.2f} grip {s.obs[103]:.2f} base {np.round(s.obs[93:96],3)}")
