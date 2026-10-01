"""Focused black-box probes for ClutteredRetrieval2DEnv dynamics."""
import math
import numpy as np
from env_client import make_env


def objects(state):
    out = {}
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        typ = getattr(obj, "type", None)
        out[name] = (obj, typ)
    return out


def vals(state, name, fields):
    obj = state.get_object_from_name(name)
    return [float(state.get(obj, f)) for f in fields]


def initial(seed, count=None):
    env = make_env()
    opts = None if count is None else {"object_count": count}
    state, info = env.reset(seed=seed, options=opts)
    print("seed", seed, "count", count, "names", state.get_object_names(), "info", info)
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        fs = ["x", "y", "theta"]
        if name == "robot": fs += ["base_radius", "arm_joint", "arm_length", "vacuum", "gripper_height", "gripper_width"]
        else: fs += ["static", "width", "height"]
        print(name, vals(state, name, fs))
    env.close()


def action_response(seed, action, n=1, count=0):
    env = make_env()
    s, _ = env.reset(seed=seed, options={"object_count": count})
    before = vals(s, "robot", ["x", "y", "theta", "arm_joint", "arm_length", "vacuum"])
    for _ in range(n):
        s, r, term, trunc, info = env.step(np.asarray(action, dtype=np.float32))
    after = vals(s, "robot", ["x", "y", "theta", "arm_joint", "arm_length", "vacuum"])
    print("response", seed, action, n, "before", before, "after", after, "rtt", r, term, trunc)
    env.close()


def drive(env, state, x, y, theta=None, arm=None, vac=0.0, limit=200):
    for _ in range(limit):
        rx, ry, rt, aj = vals(state, "robot", ["x", "y", "theta", "arm_joint"])
        dx, dy = x-rx, y-ry
        dt = 0.0 if theta is None else (theta-rt+math.pi) % (2*math.pi)-math.pi
        da = 0.0 if arm is None else arm-aj
        a = np.array([np.clip(dx,-.05,.05), np.clip(dy,-.05,.05),
                      np.clip(dt,-.19634954,.19634954), np.clip(da,-.1,.1), vac], np.float32)
        state, r, term, trunc, info = env.step(a)
        if max(abs(dx),abs(dy),abs(dt),abs(da)) < 1e-4 or term or trunc:
            break
    return state, term


def grasp_distance(seed, distance):
    env = make_env(); s, _ = env.reset(seed=seed, options={"object_count": 0})
    bx, by = vals(s, "target_block", ["x", "y"])
    th = vals(s, "robot", ["theta"])[0]
    tx, ty = bx-distance*math.cos(th), by-distance*math.sin(th)
    s, term = drive(env,s,tx,ty,theta=th,arm=.1,vac=0)
    pre = vals(s,"target_block",["x","y","theta"]); rob=vals(s,"robot",["x","y","theta","arm_joint"])
    s,_,_,_,_=env.step(np.array([0,0,0,0,1],np.float32))
    contact = vals(s,"target_block",["x","y","theta"])
    s,_,_,_,_=env.step(np.array([0,.04,0,0,1],np.float32))
    post = vals(s,"target_block",["x","y","theta"])
    print("graspD",distance,"rob",rob,"block",pre,"contact",contact,"post",post,"delta",[post[i]-contact[i] for i in range(3)])
    env.close()


def centered_grasp(seed, base_distance, arm_target):
    env=make_env(); s,_=env.reset(seed=seed,options={"object_count":0})
    bx,by=vals(s,"target_block",["x","y"]); rx,ry=vals(s,"robot",["x","y"])
    ang=math.atan2(by-ry,bx-rx)
    tx,ty=bx-base_distance*math.cos(ang),by-base_distance*math.sin(ang)
    s,_=drive(env,s,tx,ty,theta=ang,arm=.1,vac=0)
    parked=vals(s,"robot",["x","y","theta","arm_joint"])
    s,_=drive(env,s,tx,ty,theta=ang,arm=arm_target,vac=1)
    reached=vals(s,"robot",["x","y","theta","arm_joint"]); pre=vals(s,"target_block",["x","y","theta"])
    s,_,_,_,_=env.step(np.array([0,.04,0,0,1],np.float32)); post=vals(s,"target_block",["x","y","theta"])
    print("center",base_distance,arm_target,"park",parked,"reached",reached,"blockdelta",[post[i]-pre[i] for i in range(3)])
    env.close()


def incremental_grasp(seed, base_distance):
    env=make_env(); s,_=env.reset(seed=seed,options={"object_count":0})
    bx,by=vals(s,"target_block",["x","y"]); rx,ry=vals(s,"robot",["x","y"])
    ang=math.atan2(by-ry,bx-rx); tx=bx-base_distance*math.cos(ang); ty=by-base_distance*math.sin(ang)
    s,_=drive(env,s,tx,ty,theta=ang,arm=.1,vac=0)
    s,_,_,_,_=env.step(np.array([0,0,0,0,1],np.float32))
    rows=[]
    for i in range(30):
        old=vals(s,"target_block",["x","y"]); oldj=vals(s,"robot",["arm_joint"])[0]
        s,_,_,_,_=env.step(np.array([0,0,0,.005,1],np.float32))
        new=vals(s,"target_block",["x","y"]); newj=vals(s,"robot",["arm_joint"])[0]
        if newj == oldj or new != old: rows.append((i,oldj,newj,[new[k]-old[k] for k in range(2)]))
    pre=vals(s,"target_block",["x","y"]); s,_,_,_,_=env.step(np.array([0,.04,0,0,1],np.float32)); post=vals(s,"target_block",["x","y"])
    print("incremental",base_distance,"events",rows,"move_delta",[post[k]-pre[k] for k in range(2)])
    env.close()


if __name__ == "__main__":
    for sd in range(3): initial(sd, 0)
    for a in ([.05,0,0,0,0], [0,.05,0,0,0], [0,0,.19634954,0,0], [0,0,0,.1,0], [0,0,0,-.1,0]):
        action_response(0, a)
    for d in (.17,.20,.22,.235,.25,.27,.30,.33): grasp_distance(0,d)
    for aj in (.1,.15,.2,.215,.25,.3,.4,.5): centered_grasp(0,.4,aj)
    for d in (.25,.28,.30,.32,.34,.36,.38,.40): incremental_grasp(0,d)
