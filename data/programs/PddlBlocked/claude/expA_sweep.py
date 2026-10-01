import numpy as np, fk, json, sys
from env_client import make_env
from lib_util import robot, step_to

BASE = np.array([3.72, 0.10, 0.0])
R = fk.grasp_R(0.0)

class Rig:
    def __init__(self):
        self.env = make_env(); self.reset()
    def reset(self):
        obs,_ = self.env.reset(seed=1)
        self.blk = obs.data[obs.get_object_from_name("blocker")][:3].copy()
        obs,rej,n = step_to(self.env, obs, BASE, robot(obs)[3:10])
        self.obs = obs
        self.home = robot(obs)[3:10].copy()
        self.q_cur = self.home.copy()
    def trial(self, a, b, c):
        blk = self.blk
        tgt = np.array([blk[0]-a, blk[1]+b, blk[2]+c])
        pre = tgt - np.array([0.15,0,0])
        q_pre,e1 = fk.ik(pre, R, BASE, self.q_cur, seeds=6)
        q_g,e2 = fk.ik(tgt, R, BASE, q_pre, seeds=6)
        rec = dict(a=round(a,3),b=round(b,3),c=round(c,3),ik_pre=round(float(e1),4),ik=round(float(e2),4))
        if e2 > 0.01:
            rec['status']='ik_fail'; return rec
        # open first
        z=np.zeros(11,dtype=np.float32); z[10]=1.0
        self.obs,_,_,_,_ = self.env.step(z)
        o,rj1,n1 = step_to(self.env, self.obs, BASE, q_pre); self.obs=o
        o,rj2,n2 = step_to(self.env, self.obs, BASE, q_g); self.obs=o
        qa = robot(self.obs)[3:10]
        perr = float(np.linalg.norm(fk.pose_err(qa, BASE, tgt, R)[:3]))
        rec['rej_pre']=int(rj1); rec['rej_g']=int(rj2); rec['reach_err']=round(perr,4)
        z=np.zeros(11,dtype=np.float32); z[10]=-1.0
        self.obs,_,_,_,_ = self.env.step(z)
        ga = float(robot(self.obs)[11])
        rec['ga']=int(ga>0.5)
        rec['status']='ok' if not (rj1 or rj2) else 'rejected'
        # release + retract
        z=np.zeros(11,dtype=np.float32); z[10]=1.0
        self.obs,_,_,_,_ = self.env.step(z)
        o,_,_ = step_to(self.env, self.obs, BASE, q_pre); self.obs=o
        self.q_cur = q_pre.copy()
        nb = self.obs.data[self.obs.get_object_from_name("blocker")][:3]
        rec['blk_moved']=round(float(np.linalg.norm(nb-blk)),4)
        if rec['blk_moved']>0.01:
            self.reset()
        return rec

rig = Rig()
print("blocker", rig.blk, flush=True)
out=[]
def run(name, cands):
    print("===",name, flush=True)
    for (a,b,c) in cands:
        r = rig.trial(a,b,c); r['sweep']=name; out.append(r)
        print(json.dumps(r), flush=True)

A=[round(x,3) for x in np.arange(-0.06,0.1001,0.02)]
B=[round(x,3) for x in np.arange(-0.06,0.0601,0.02)]
C=[round(x,3) for x in np.arange(-0.10,0.1001,0.02)]
run("A",[(a,0.0,0.0) for a in A])
run("B",[(0.03,b,0.0) for b in B])
run("C",[(0.03,0.0,c) for c in C])
json.dump(out, open("expA_sweep1.json","w"))
rig.env.close()
