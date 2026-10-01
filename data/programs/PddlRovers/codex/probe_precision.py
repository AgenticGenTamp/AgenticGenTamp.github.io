import numpy as np
from env_client import make_env


def vals(s, sp, name):
    o=s.get_object_from_name(name)
    return {f:float(s.get(o,f)) for f in sp.type_features[o.type]}


def action(i, dx=0, dy=0, dt=0, sel=0):
    a=np.zeros(8,np.float32); j=4*i; a[j:j+4]=(dx,dy,dt,sel); return a


def move(env,s,i,x,y,order="xy"):
    sp=env.observation_space
    for axis in order:
        for _ in range(40):
            q=vals(s,sp,f"rover{i}"); v=(x-q['x']) if axis=='x' else (y-q['y'])
            if abs(v)<2e-6: break
            v=max(-.2,min(.2,v)); kw={'dx':v} if axis=='x' else {'dy':v}
            s,*_=env.step(action(i,**kw))
    return s


def boundary():
  for x in [2.30,2.34,2.349,2.35,2.351,2.36,2.40]:
    e=make_env();s,_=e.reset(seed=0);s=move(e,s,0,x,-1.75);print('BOUND',x,vals(s,e.observation_space,'rover0')['x']);e.close()


def wall():
  # Move upward at x=.6, then try to cross wall to x=-.6.
  for y in [-2.3,-2.2,-2.1,-2.0,-1.9,-1.8,-1.7,-1.6,-1.5,1.45,1.5,1.55,1.6,1.65,1.7,1.75]:
    e=make_env();s,_=e.reset(seed=0);s=move(e,s,0,.6,y,'xy');s=move(e,s,0,-.6,y,'xy');print('WALL',y,vals(s,e.observation_space,'rover0'));e.close()

def wall2():
  for y in [-2.3,-2.1,-1.9,-1.7,-1.5,-1.0,-.5,0,.5,.8,1.0,1.2,1.4,1.6]:
    e=make_env();s,_=e.reset(seed=0)
    s=move(e,s,0,1.2,y,'xy'); at=vals(s,e.observation_space,'rover0')
    s=move(e,s,0,-1.2,y,'xy'); print('WALL2',y,'at',at['x'],at['y'],'end',vals(s,e.observation_space,'rover0')['x']);e.close()

def boundary2():
  for x in [2.201,2.21,2.22,2.24,2.249,2.25,2.251,2.26,2.28,2.299]:
    e=make_env();s,_=e.reset(seed=0);s=move(e,s,0,x,-1.75);print('BOUND2',x,vals(s,e.observation_space,'rover0')['x']);e.close()

def home_threshold():
  for d in [.249,.25,.251,.30,.2*2**.5]:
    e=make_env();s,_=e.reset(seed=0);sp=e.observation_space
    if d==.2*2**.5: s,*_=e.step(action(0,dx=.2,dy=.2))
    else: s=move(e,s,0,1+d,-1.75)
    print('HOME',d,vals(s,sp,'rover0'));e.close()

def boundary3():
  for x in [2.265,2.269,2.27,2.271,2.275,2.279]:
    e=make_env();s,_=e.reset(seed=0);s=move(e,s,0,x,-1.75);print('BOUND3',x,vals(s,e.observation_space,'rover0')['x']);e.close()

def yboundary():
  for y in [-2.151,-2.17,-2.18,-2.19,-2.2,-2.21,-2.22,-2.23,-2.24,-2.25]:
    e=make_env();s,_=e.reset(seed=0);s=move(e,s,0,1.2,y,'xy');print('YBOUND',y,vals(s,e.observation_space,'rover0')['y']);e.close()

def occlusion():
  # objective2 -> pillar7 line; camera just behind pillar, with perpendicular offsets.
  obj=np.array([-1.8657476902,2.1903953552]); pil=np.array([-1.3539739847,.5535165071])
  u=pil-obj; u=u/np.linalg.norm(u); perp=np.array([-u[1],u[0]])
  base=pil+.18*u
  for off in [0,.03,.05,.07,.1,.15]:
    p=base+off*perp;e=make_env();s,_=e.reset(seed=0);sp=e.observation_space
    s=move(e,s,1,float(p[0]),float(p[1]),'yx');before=vals(s,sp,'rover1');s,*_=e.step(action(1,sel=-.5))
    print('OCC',off,'target',p,'actual',before['x'],before['y'],'dist',np.linalg.norm(p-obj),'cal',vals(s,sp,'rover1')['calibrated']);e.close()

def rejected_operator():
  e=make_env();s,_=e.reset(seed=0);sp=e.observation_space;q=vals(s,sp,'sample4')
  s=move(e,s,1,q['x'],q['y']-.2,'xy'); before=vals(s,sp,'rover1')
  s,*_=e.step(action(1,dy=-.2,sel=-5/6));print('REJECTOP before',before,'after',vals(s,sp,'rover1'),vals(s,sp,'sample4'));e.close()

def yboundary2():
  for y in [-2.26,-2.27,-2.271,-2.275,-2.28,-2.29,-2.30,-2.31,-2.32]:
    e=make_env();s,_=e.reset(seed=0);s=move(e,s,0,1.2,y,'xy');print('YBOUND2',y,vals(s,e.observation_space,'rover0')['y']);e.close()

def pillar_shape():
  px,py=.654291749,-.198718399
  for dy in [0,.1,.2,.25,.27,.28,.30]:
    e=make_env();s,_=e.reset(seed=0);sp=e.observation_space;s=move(e,s,0,1.2,py+dy,'xy')
    for _ in range(100):
      old=vals(s,sp,'rover0');s,*_=e.step(action(0,dx=-.01));new=vals(s,sp,'rover0')
      if old['x']==new['x']:break
    print('PILLAR',dy,vals(s,sp,'rover0')['x'],vals(s,sp,'rover0')['y']);e.close()

def candidates():
  for seed in range(10):
    e=make_env();s,_=e.reset(seed=seed);sp=e.observation_space
    os=[(o.name,vals(s,sp,o.name)) for o in s.get_objects(sp.get_type('objective'))]
    ps=[(o.name,vals(s,sp,o.name)) for o in s.get_objects(sp.get_type('obstacle')) if vals(s,sp,o.name)['half_x']<.1]
    for on,o in os:
      for pn,p in ps:
        ov=np.array([o['x'],o['y']]);pv=np.array([p['x'],p['y']]);d=np.linalg.norm(pv-ov)
        if .3<d<1.5:
          cam=pv+.35*(pv-ov)/d
          if max(abs(cam))<2.1 and ((cam[0]>.3) == (o['x']>.3) or (cam[0]<-.3)==(o['x']<-.3)):
            print(seed,on,pn,'d',round(d,3),'cam',cam)
    e.close()

def occlusion2():
  seed=2
  for off in [0,.03,.06,.09,.12,.2]:
    e=make_env();s,_=e.reset(seed=seed);sp=e.observation_space;o=vals(s,sp,'objective0');p=vals(s,sp,'obstacle11')
    ov=np.array([o['x'],o['y']]);pv=np.array([p['x'],p['y']]);u=(pv-ov)/np.linalg.norm(pv-ov);perp=np.array([-u[1],u[0]])
    cam=pv+.35*u+off*perp;s=move(e,s,0,float(cam[0]),float(cam[1]),'yx');before=vals(s,sp,'rover0');s,*_=e.step(action(0,sel=-.5))
    print('OCC2',off,'cam',cam,'actual',before['x'],before['y'],'cal',vals(s,sp,'rover0')['calibrated']);e.close()


def sample_threshold():
  # rover1 approaches soil sample4 at bottom from west, operator simultaneous on final x move
  for dist in [.249,.25,.251,.26,.30]:
    e=make_env();s,_=e.reset(seed=0);sp=e.observation_space; q=vals(s,sp,'sample4')
    s=move(e,s,1,q['x']-dist-.1,q['y'],'yx')
    s,*_=e.step(action(1,dx=.1,sel=-5/6))
    print('SAMPLE',dist,vals(s,sp,'rover1'), vals(s,sp,'sample4'));e.close()


def ordering():
  e=make_env();s,_=e.reset(seed=0);sp=e.observation_space;q=vals(s,sp,'sample4')
  # Arrange rover1 0.35 west, then move 0.11 and sample => final distance .24 only.
  s=move(e,s,1,q['x']-.35,q['y'],'yx'); print('ORDERBEFORE',vals(s,sp,'rover1'))
  s,*_=e.step(action(1,dx=.11,sel=-5/6)); print('ORDERAFTER',vals(s,sp,'rover1'),vals(s,sp,'sample4'));e.close()


def heading():
  # Position on right with visible objective(s), test calibration after arbitrary rotations.
  for th in [-3.0,-1.5,0,1.5,3.0]:
    e=make_env();s,_=e.reset(seed=0);sp=e.observation_space;s=move(e,s,0,.4,.97,'yx')
    q=vals(s,sp,'rover0'); delta=((th-q['theta']+np.pi)%(2*np.pi))-np.pi
    while abs(delta)>1e-5:
      z=max(-.4,min(.4,delta));s,*_=e.step(action(0,dt=z));delta-=z
    s,*_=e.step(action(0,sel=-.5));print('HEAD',th,vals(s,sp,'rover0'));e.close()

def image_range():
  o=(1.9636701345443726,2.109961986541748)
  for dist in [1.99,2.0,2.001,2.01,2.1]:
    e=make_env();s,_=e.reset(seed=0);sp=e.observation_space
    s=move(e,s,0,o[0],o[1]-dist,'xy'); before=vals(s,sp,'rover0')
    s,*_=e.step(action(0,sel=-.5));print('IMAGE_RANGE',dist,before,vals(s,sp,'rover0')['calibrated']);e.close()

def send_basic():
  e=make_env();s,_=e.reset(seed=0);sp=e.observation_space;q=vals(s,sp,'sample4')
  s=move(e,s,1,q['x']-.2,q['y'],'yx');s,*_=e.step(action(1,sel=-5/6));
  print('SEND acquired',vals(s,sp,'rover1'),vals(s,sp,'sample4'))
  s,*_=e.step(action(1,sel=.5));print('SEND result',vals(s,sp,'sample4'));e.close()

def wallfine():
  for y in [-1.5,-1.0,-.5,.2,.5,.8,1.0,1.2,1.4]:
    e=make_env();s,_=e.reset(seed=0);sp=e.observation_space;s=move(e,s,0,1.2,y,'xy')
    rejects=0
    for _ in range(80):
      old=vals(s,sp,'rover0');s,*_=e.step(action(0,dx=-.02));new=vals(s,sp,'rover0')
      if old['x']==new['x']: rejects+=1; break
    print('WALLFINE',y,'x',vals(s,sp,'rover0')['x'],'reject',rejects);e.close()

def wallfine2():
  for y in [.5,.7,.8,.85,.9,.95,1.0,1.05,1.1,1.2,1.4,1.5,1.6]:
    e=make_env();s,_=e.reset(seed=2);sp=e.observation_space;s=move(e,s,0,1.2,y,'xy')
    for _ in range(100):
      old=vals(s,sp,'rover0');s,*_=e.step(action(0,dx=-.02));new=vals(s,sp,'rover0')
      if old['x']==new['x']: break
    print('WALLFINE2',y,'starty',vals(s,sp,'rover0')['y'],'x',vals(s,sp,'rover0')['x']);e.close()


if __name__=='__main__':
 import sys
 globals()[sys.argv[1]]()
