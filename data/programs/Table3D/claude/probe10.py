import numpy as np
from env_client import make_env
from ik2 import solve_pos_axis
J=["joint_%d"%i for i in range(1,8)]
BZ=0.4
CF=["pose_x","pose_y","pose_z","grasp_active"]
def getq(o):
    r=o.get_object_from_name("robot"); return np.array([float(o.get(r,f)) for f in J])
def dump(o,tag):
    s=[]
    for n in sorted(o.get_object_names()):
        ob=o.get_object_from_name(n)
        if n=="robot":
            s.append("robot fs=%.2f ga=%.2f gtf=%s"%(float(o.get(ob,"finger_state")),float(o.get(ob,"grasp_active")),
              [round(float(o.get(ob,f)),3) for f in ["grasp_tf_x","grasp_tf_y","grasp_tf_z","grasp_tf_qw"]]))
        elif n.startswith("cube"):
            s.append("%s %s"%(n,[round(float(o.get(ob,f)),3) for f in CF]))
    print(tag,"|".join(s))
env=make_env(); obs,_=env.reset(seed=0); q=getq(obs)
c=obs.get_object_from_name("cube0")
cp=np.array([float(obs.get(c,f)) for f in ["pose_x","pose_y","pose_z"]])-np.array([0,0,BZ])
def goto(qt,lim=0.3):
    global q,obs
    rej=0
    for _ in range(120):
        d=qt-q
        if np.max(np.abs(d))<3e-3: return True
        a=np.zeros(11); a[3:10]=np.clip(d,-lim,lim)
        o2,*_=env.step(a); qn=getq(o2)
        if np.allclose(qn,q,atol=1e-9):
            rej+=1
            if rej>2: return False
        q=qn; obs=o2
    return False
qt,pe,ae=solve_pos_axis(cp+np.array([0,0,0.20]),0.0)
print("ik",pe,ae, goto(qt))
dump(obs,"at pose")
for v in [1.0,-1.0]:
    a=np.zeros(11); a[10]=v; obs,rw,t,tr,i=env.step(a); dump(obs,"grip %.1f rw=%s t=%s"%(v,rw,t))
# lift
qt2,pe,ae=solve_pos_axis(cp+np.array([0,0,0.35]),0.0,q0=q)
goto(qt2); dump(obs,"lifted")
env.close()
