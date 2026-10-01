import sys, json
import numpy as np
from env_client import make_env
import kinova_fk as K

FLAG = float(sys.argv[1])
np.set_printoptions(precision=4, suppress=True)

env = make_env()
obs,_ = env.reset(seed=0)
obs = np.asarray(obs,float)
base = obs[93:96].copy()
q0 = obs[96:103].copy()
objA = obs[0:3].copy()
print("EE at reset (fk):", K.fk_position(q0, base), " objA:", objA)

def step(a):
    global obs
    o,r,te,tr,i = env.step(np.asarray(a,float))
    obs = np.asarray(o,float)
    return r

# phase 0: open gripper (both conditions start at 0 -> use opposite of FLAG then set FLAG)
a = np.zeros(11); a[10] = 1.0-FLAG
for _ in range(5): step(a)

def goto(target, grip, nsteps=80, tag=""):
    for t in range(nsteps):
        q = obs[96:103]
        qt, ok, inf = K.ik(target, q_init=q, base_pose=obs[93:96], restarts=2)
        dq = np.clip(qt - q, -0.05, 0.05)
        a = np.zeros(11); a[3:10] = dq*4.0; a[10] = grip
        step(a)
    ee = K.fk_position(obs[96:103], obs[93:96])
    print(tag, "EE", ee, "err to target", np.linalg.norm(ee-target))
    return ee

# approach above object then down
pre = objA + np.array([0,0,0.12])
goto(pre, 1.0-FLAG, 70, "pre")
print("  objA now", obs[0:3], "vel", obs[7:10])
goto(objA, 1.0-FLAG, 60, "at")
print("  objA now", obs[0:3], "vel", obs[7:10])

# set gripper flag
a = np.zeros(11); a[10]=FLAG
for _ in range(15): step(a)
print("after grip flag=%.1f obs103=%.3f objA %s"%(FLAG, obs[103], obs[0:3]))

# lift: move EE up 0.25
tgt = objA + np.array([0,0,0.25])
goto(tgt, FLAG, 80, "lift")
print("after lift objA", obs[0:3], "base", obs[93:96])

# drive base -x
b0 = obs[93].copy(); o0 = obs[0:3].copy()
a = np.zeros(11); a[0]=-0.5; a[10]=FLAG
for _ in range(20): step(a)
print("BASE dx=%.4f  objA d=%s"%(obs[93]-b0, obs[0:3]-o0))
print("RESULT flag=%.1f  obj_tracks_base=%s"%(FLAG, abs((obs[0:3]-o0)[0]-(obs[93]-b0))<0.3))
print("final objA", obs[0:3], "objB", obs[16:19], "objC", obs[32:35])
env.close()
