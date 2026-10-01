import math
import numpy as np
from env_client import make_env


def get(s, n, f): return float(s.get(s.get_object_from_name(n), f))
def lim(x, m): return max(-m, min(m, x))
def adiff(a): return (a + math.pi) % (2*math.pi)-math.pi


def run(seed):
    env=make_env(); s,info=env.reset(seed=seed)
    h0x=get(s,'hook','x')
    targets=[
      # x,y,theta,arm,gap,dwell
      (h0x,.75,-math.pi/2,.4,.25,8),
      (h0x,.75,-math.pi/2,.4,.08,12),
      (h0x,2.45,-math.pi/2,.4,.08,5),
      (.55,2.45,-math.pi/2,.4,.08,5),
      (.55,.78,-math.pi/2,.4,.08,5),
      (1.48,.78,-math.pi/2,.4,.08,10),
      (1.48,2.45,-math.pi/2,.4,.08,5),
      (3.1,2.45,-math.pi/2,.4,.08,10),
    ]
    stage=dwell=0
    initial=[n for n in s.get_object_names() if n.startswith('small_')]
    for t in range(1000):
      x,y,th,arm,gap,need=targets[stage]
      rx,ry,rt=get(s,'robot','x'),get(s,'robot','y'),get(s,'robot','theta')
      err=(abs(x-rx),abs(y-ry),abs(adiff(th-rt)),abs(arm-get(s,'robot','arm_length')),abs(gap-get(s,'robot','finger_gap')))
      arrived=max(err[0]/.03,err[1]/.03,err[2]/.098,err[3]/.08,err[4]/.015)<1.2
      if arrived: dwell+=1
      else: dwell=0
      if dwell>=need and stage<len(targets)-1:
        stage+=1; dwell=0; x,y,th,arm,gap,need=targets[stage]
      a=np.array([lim(x-rx,.03),lim(y-ry,.03),lim(adiff(th-rt),.098),lim(arm-get(s,'robot','arm_length'),.08),lim(gap-get(s,'robot','finger_gap'),.015)])
      s,r,term,trunc,inf=env.step(a)
      if t%40==0 or term:
        objs=[n for n in s.get_object_names() if n.startswith('small_')]
        right=sum(get(s,n,'x')>2 for n in objs)
        print(seed,t,'st',stage,'rob',round(get(s,'robot','x'),2),round(get(s,'robot','y'),2),'hook',round(get(s,'hook','x'),2),round(get(s,'hook','y'),2),round(get(s,'hook','theta'),2),get(s,'hook','held'),'right',right,'/',len(objs))
      if term or trunc: break
    env.close(); return term,t


print('RESULT', 0, run(0))
