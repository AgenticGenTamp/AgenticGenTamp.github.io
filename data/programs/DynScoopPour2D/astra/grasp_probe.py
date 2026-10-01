from env_client import make_env
import numpy as np, math, concurrent.futures

def trial(off,yoff,theta):
 e=make_env();s,_=e.reset(seed=42)
 r=s.get_objects(e.observation_space.get_type('kin_robot'))[0];h=s.get_objects(e.observation_space.get_type('hook'))[0]
 hx,hy=s.get(h,'x'),s.get(h,'y')
 target=(hx-off*math.cos(theta),hy+yoff-off*math.sin(theta),theta,.4,.25)
 def move(t,n):
  nonlocal s
  for k in range(n):
   vals=[s.get(r,f) for f in ('x','y','theta','arm_length','finger_gap')];d=np.array(t)-vals;d[2]=(d[2]+math.pi)%(2*math.pi)-math.pi
   if max(abs(d))<1e-4:break
   s,*_=e.step(np.clip(d,[-.0299,-.0299,-.0979,-.0799,-.0149],[.0299,.0299,.0979,.0799,.0149]))
  return s.get(h,'held')
 move(target,140)
 t=list(target);t[-1]=0
 held=move(t,30)
 result=(off,yoff,theta,held,[round(s.get(r,f),3) for f in ('x','y','theta','arm_length','finger_gap')],[round(s.get(h,f),3) for f in ('x','y','theta')])
 e.close();print(result,flush=True);return result
if __name__=='__main__':
 args=[(off,yo,t) for off in (.5,.6,.7,.8) for yo,t in ((.45,0),(.25,0),(.45,-math.pi/2))]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex: list(ex.map(lambda a:trial(*a),args))
