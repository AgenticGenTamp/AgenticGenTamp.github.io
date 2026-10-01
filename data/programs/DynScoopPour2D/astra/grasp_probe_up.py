from env_client import make_env
import numpy as np,math,concurrent.futures

def trial(xoff,yoff,t):
 e=make_env();s,_=e.reset(seed=42);r=s.get_objects(e.observation_space.get_type('kin_robot'))[0];h=s.get_objects(e.observation_space.get_type('hook'))[0]
 hx,hy=s.get(h,'x'),s.get(h,'y')
 def move(v,n):
  nonlocal s
  for _ in range(n):
   d=np.array(v)-[s.get(r,f) for f in ('x','y','theta','arm_length','finger_gap')];d[2]=(d[2]+math.pi)%(2*math.pi)-math.pi
   if max(abs(d))<1e-4:break
   s,*_=e.step(np.clip(d,[-.0299,-.0299,-.0979,-.0799,-.0149],[.0299,.0299,.0979,.0799,.0149]))
 move((2.8,.8,t,.4,.25),140)
 move((hx+xoff,hy+yoff,t,.4,.25),80)
 move((hx+xoff,hy+yoff,t,.4,0),25)
 print((xoff,yoff,t,s.get(h,'held'),[round(s.get(r,f),3) for f in ('x','y','theta','arm_length','finger_gap')],[round(s.get(h,f),3) for f in ('x','y','theta')]),flush=True)
 e.close()
if __name__=='__main__':
 args=[(x,y,t) for x,y,t in [(-.1,.2,math.pi/2),(-.15,.2,math.pi/2),(-.1,.25,math.pi/2),(-.3,.5,0),(-.32,.48,0),(-.32,.55,0),(-.32,.0,0),(-.32,.05,0)]]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(lambda a:trial(*a),args))
