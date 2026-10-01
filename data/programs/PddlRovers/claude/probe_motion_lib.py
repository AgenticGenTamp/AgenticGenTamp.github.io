import numpy as np
from env_client import make_env
NOOP = 0.1667

def new_env(seed=0):
    env = make_env()
    obs, info = env.reset(seed=seed)
    return env, obs, info

def pose(obs, i=0):
    o = obs.get_object_from_name(f"rover{i}")
    return np.array([float(obs.get(o,'x')), float(obs.get(o,'y')), float(obs.get(o,'theta'))])

def act(dx=0.0, dy=0.0, dth=0.0, i=0):
    a = np.zeros(8, dtype=np.float32)
    a[3] = NOOP; a[7] = NOOP
    b = 4*i
    a[b+0]=dx; a[b+1]=dy; a[b+2]=dth
    return a

def step(env, dx=0.0, dy=0.0, dth=0.0, i=0):
    obs, r, te, tr, info = env.step(act(dx,dy,dth,i))
    return obs

def try_move(env, obs, dx, dy, dth=0.0, i=0):
    """returns (newobs, moved_bool, delta)"""
    p0 = pose(obs, i)
    obs2 = step(env, dx, dy, dth, i)
    p1 = pose(obs2, i)
    d = p1-p0
    moved = (abs(d[0])>1e-6) or (abs(d[1])>1e-6)
    return obs2, moved, d

def goto(env, obs, tx, ty, i=0, tol=0.003, maxit=400):
    """greedily move axis-wise to target; returns obs"""
    for _ in range(maxit):
        p = pose(obs,i)
        ex, ey = tx-p[0], ty-p[1]
        if abs(ex)<tol and abs(ey)<tol: break
        dx = float(np.clip(ex, -0.2, 0.2)); dy = float(np.clip(ey, -0.2, 0.2))
        obs, moved, d = try_move(env, obs, dx, dy, 0.0, i)
        if not moved:
            # try one axis at a time
            obs, m1, _ = try_move(env, obs, dx, 0.0, 0.0, i)
            obs, m2, _ = try_move(env, obs, 0.0, dy, 0.0, i)
            if not m1 and not m2:
                return obs, False
    return obs, True

def push_dir(env, obs, ux, uy, i=0, coarse=0.2, fine=0.001, limit=400):
    """Move in direction (ux,uy) (unit) until blocked, refining step size. Returns final obs,pos."""
    step_size = coarse
    it=0
    while step_size >= fine and it < limit:
        obs, moved, d = try_move(env, obs, ux*step_size, uy*step_size, 0.0, i)
        it+=1
        if not moved:
            step_size = step_size/2.0
    return obs, pose(obs,i)

import random
def nav(env, obs, tx, ty, i=0, tol=0.01, maxit=600, seed=1):
    rng = random.Random(seed)
    for _ in range(maxit):
        p = pose(obs,i)
        ex, ey = tx-p[0], ty-p[1]
        if abs(ex)<tol and abs(ey)<tol: return obs, True
        dx = float(np.clip(ex,-0.2,0.2)); dy = float(np.clip(ey,-0.2,0.2))
        obs,m,_ = try_move(env,obs,dx,dy,0.0,i)
        if m: continue
        # axis moves, bigger error first
        order = [(dx,0.0),(0.0,dy)] if abs(ex)>=abs(ey) else [(0.0,dy),(dx,0.0)]
        done=False
        for (a,b) in order:
            obs,m,_ = try_move(env,obs,a,b,0.0,i)
            if m: done=True; break
        if done: continue
        # sidestep perpendicular
        for _ in range(3):
            s = 0.2 if rng.random()<0.5 else -0.2
            if abs(ex)>=abs(ey): cand=(0.0,s)
            else: cand=(s,0.0)
            obs,m,_ = try_move(env,obs,cand[0],cand[1],0.0,i)
            if m: break
    p=pose(obs,i)
    return obs, (abs(p[0]-tx)<tol and abs(p[1]-ty)<tol)

def setth(env, obs, th, i=0):
    for _ in range(60):
        d=th-pose(obs,i)[2]
        if abs(d)<0.0005: break
        obs,m,dd=try_move(env,obs,0,0,float(np.clip(d,-0.4,0.4)),i)
    return obs
