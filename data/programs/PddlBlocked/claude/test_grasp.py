import numpy as np, fk
from env_client import make_env

def robot(obs):
    return obs.data[obs.get_object_from_name("robot")]

def step_to(env, obs, base_t, q_t, grip=0, maxsteps=60, verbose=False):
    rej=0
    for k in range(maxsteps):
        r = robot(obs)
        b = r[:3]; q = r[3:10]
        db = np.array(base_t) - b
        db[2] = (db[2]+np.pi)%(2*np.pi)-np.pi
        dq = np.array(q_t) - q
        for i in fk.CONT: dq[i] = (dq[i]+np.pi)%(2*np.pi)-np.pi
        a = np.zeros(11, dtype=np.float32)
        a[:3] = np.clip(db, -0.2, 0.2)
        a[3:10] = np.clip(dq, -0.2, 0.2)
        a[10] = grip
        if np.all(np.abs(a[:10])<1e-4): break
        prev = r.copy()
        obs,_,term,_,_ = env.step(a)
        if np.allclose(robot(obs)[:10], prev[:10], atol=1e-6):
            rej+=1
            if verbose: print("  rejected at step",k)
            if rej>3: break
        else: rej=0
    return obs

env = make_env()
obs,_ = env.reset(seed=1)
blk = obs.data[obs.get_object_from_name("blocker")][:3]
print("blocker", blk)
base = np.array([3.72, 0.10, 0.0])
obs = step_to(env, obs, base, robot(obs)[3:10])
print("base now", robot(obs)[:3])
R = fk.grasp_R(0.0)
results=[]
q_cur = robot(obs)[3:10]
for dz in [-0.05,0.0,0.05,0.1,0.15]:
    for dx in [-0.06,-0.03,0.0]:
        tgt = np.array([blk[0]+dx, blk[1], blk[2]+dz])
        # pregrasp
        q_pre,e1 = fk.ik(tgt-np.array([0.15,0,0]), R, base, q_cur, seeds=6)
        q_g,e2 = fk.ik(tgt, R, base, q_pre, seeds=6)
        obs = step_to(env, obs, base, q_pre)
        obs = step_to(env, obs, base, q_g)
        qa = robot(obs)[3:10]
        perr = np.linalg.norm(fk.pose_err(qa, base, tgt, R)[:3])
        a=np.zeros(11,dtype=np.float32); a[10]=-1.0
        obs,_,term,_,_ = env.step(a)
        ga = robot(obs)[11]
        results.append((dz,dx,round(e1,3),round(e2,3),round(float(perr),3),ga))
        print(results[-1])
        if ga>0.5:
            print("GRASPED! block pose", obs.data[obs.get_object_from_name("blocker")])
            print("grasp_tf", robot(obs)[12:])
            break
        a=np.zeros(11,dtype=np.float32); a[10]=1.0
        obs,_,_,_,_ = env.step(a)
    else:
        continue
    break
env.close()
