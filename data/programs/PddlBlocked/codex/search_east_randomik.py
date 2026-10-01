"""Numerical/randomized north-side blocker grasp search for seeds 34/65."""
import math
import sys
import numpy as np
from scipy.optimize import least_squares
from env_client import make_env
from probe_grasp import fk, robot_vals

LO=np.array([-.715,-.524,-.8,-2.321,-math.pi,-2.094,-math.pi])
HI=np.array([2.285,1.396,3.9,0,math.pi,0,math.pi])
Q0=np.array([.393,.333,0,-1.522,2.722,-1.22,-2.989])

def get(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def ik(base,p,direction,rng):
    def fun(q):
        T=fk(base,q)
        # Position, horizontal approach direction, and regularization.
        return np.r_[12*(T[:3,3]-p),4*np.cross(T[:3,0],direction),.015*(q-Q0)]
    best=None
    for k in range(4):
        x=Q0 if k==0 else rng.uniform(LO,HI)
        z=least_squares(fun,np.clip(x,LO,HI),bounds=(LO,HI),max_nfev=250)
        score=np.linalg.norm(fun(z.x))
        if best is None or score<best[0]: best=(score,z.x)
    return best

def advance(e,s,target,grip=1,limit=35):
    for _ in range(limit):
        cur=robot_vals(s)[:10]; d=target-cur
        d[2]=(d[2]+math.pi)%(2*math.pi)-math.pi
        d[7]=(d[7]+math.pi)%(2*math.pi)-math.pi
        d[9]=(d[9]+math.pi)%(2*math.pi)-math.pi
        a=np.zeros(11,np.float32);a[:10]=np.clip(d,-.12,.12);a[10]=grip
        old=cur.copy();s,*_=e.step(a)
        if get(s,'robot','grasp_active')>.5:return s,True
        if np.max(np.abs(d))<.003:return s,False
    return s,False

def test(seed,base,qpre,qgoal):
    e=make_env();s,_=e.reset(seed=seed)
    # Configure outside the table footprint, then approach from north.
    safe=np.r_[3.25,1.25,base[2],qpre]
    s,_=advance(e,s,safe,1,45)
    s,_=advance(e,s,np.r_[base,qpre],1,30)
    s,held=advance(e,s,np.r_[base,qgoal],-1,25)
    actual=robot_vals(s)[:10].copy()
    if held:
        # Demonstrate a full collision-free removal toward the north.
        lifted=qgoal.copy();lifted[1]=max(LO[1],qgoal[1]-.18)
        s,_=advance(e,s,np.r_[base,lifted],-1,15)
        out=base.copy();out[1]=1.45
        s,_=advance(e,s,np.r_[out,lifted],-1,20)
        removed=np.array([get(s,'blocker','pose_x'),get(s,'blocker','pose_y'),get(s,'blocker','pose_z')])
    else: removed=None
    e.close();return held,actual,removed

def main(seed):
    e=make_env();s,_=e.reset(seed=seed)
    real=np.array([get(s,'blocker','pose_'+x) for x in 'xyz']);e.close()
    rng=np.random.default_rng(seed+901)
    direction=np.array([0.,-1.,0.])
    count=0
    for yaw in (-.8,-.4,0,.4,.8,1.2,1.6,2.0,2.4,2.8):
      for bx in (3.4,3.55,3.7,3.85,4.0,4.15,4.3,4.45,4.6):
       for by in (1.0,1.05,1.12):
        base=np.array([bx,by,yaw])
        for calx in (-.10,-.06,-.02,.02,.06):
         # Empirical FK is about 7cm too short and 7.5cm too high.
         goal=real+np.array([calx,0,.075])
         eg,qg=ik(base,goal,direction,rng)
         ep,qp=ik(base,goal-direction*.13,direction,rng)
         if eg>.20 or ep>.20: continue
         count+=1
         held,actual,removed=test(seed,base,qp,qg)
         if count%25==0: print('tested',count,'base',base,'errors',ep,eg,flush=True)
         if held:
            print('HIT seed',seed,'base',base.tolist(),'qpre',qp.tolist(),'qgoal',qg.tolist(),
                  'actual',actual.tolist(),'removed',removed.tolist(),flush=True)
            return
    print('NONE',seed,count)

if __name__=='__main__':main(int(sys.argv[1]) if len(sys.argv)>1 else 34)
