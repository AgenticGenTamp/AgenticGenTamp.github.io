import sys
import numpy as np
from env_client import make_env

SAMPLE=-5/6; CAL=-.5; IMAGE=-1/6; NOOP=1/6; SEND=.5; DROP=5/6
env=make_env(); s,info=env.reset(seed=1)
rt=next(t for t in env.observation_space.types if t.name=='rover')
st=next(t for t in env.observation_space.types if t.name=='sample')
ot=next(t for t in env.observation_space.types if t.name=='objective')
rovers=list(s.get_objects(rt)); samples=list(s.get_objects(st)); objectives=list(s.get_objects(ot))
stepn=0
firstterm=None
def val(o,f): return float(s.get(o,f))
def act(a0=(0,0,NOOP),a1=(0,0,NOOP), label=''):
 global s,stepn,firstterm
 a=np.zeros(8,np.float32); a[0],a[1],a[3]=a0; a[4],a[5],a[7]=a1
 s,r,term,trunc,inf=env.step(a); stepn+=1
 if term and firstterm is None: firstterm=(stepn,label)
 if label:
  print(label,'n',stepn,'term',term,'info',inf,'r',[(round(val(q,'x'),2),round(val(q,'y'),2),val(q,'store_full'),val(q,'calibrated'),val(q,'at_home')) for q in rovers],
   'samp',[(x.name,val(x,'analyzed_rover0'),val(x,'analyzed_rover1'),val(x,'received_analysis')) for x in samples],
   'obj',[(x.name,val(x,'have_image_rover0'),val(x,'have_image_rover1'),val(x,'received_image')) for x in objectives])
 return term
def goto(i,x,y):
 global s
 for _ in range(100):
  rx,ry=val(rovers[i],'x'),val(rovers[i],'y'); dx=np.clip(x-rx,-.2,.2); dy=np.clip(y-ry,-.2,.2)
  if abs(dx)<.01 and abs(dy)<.01:return
  args=(float(dx),float(dy),NOOP)
  act(args if i==0 else (0,0,NOOP), args if i==1 else (0,0,NOOP))
 print('GOTO_FAIL',i,x,y,val(rovers[i],'x'),val(rovers[i],'y'))

# Simultaneously approach an easy stone (rover1 sample1) and soil (rover0 sample3).
for _ in range(4):
 p=[]
 for i,(x,y) in enumerate([(val(samples[3],'x'),val(samples[3],'y')),(val(samples[1],'x'),val(samples[1],'y'))]):
  p.append((float(np.clip(x-val(rovers[i],'x'),-.2,.2)),float(np.clip(y-val(rovers[i],'y'),-.2,.2)),NOOP))
 act(p[0],p[1])
act((0,0,SAMPLE),(0,0,SAMPLE),'sampled')
mode=sys.argv[1] if len(sys.argv)>1 else 'senddrop'
if mode=='dropsend':
 act((0,0,DROP),(0,0,DROP),'dropped-before-send'); act((0,0,SEND),(0,0,SEND),'sent-after-drop')
elif mode=='nodrop':
 act((0,0,SEND),(0,0,SEND),'sent-keeping-stores')
else:
 act((0,0,SEND),(0,0,SEND),'sent-before-drop'); act((0,0,DROP),(0,0,DROP),'dropped-after-send')

# Head around opposite ends of the central wall, image one objective each.
for i,pts in enumerate(([(2.2,-1.0),(2.2,.5),(2.1,.5)], [(-1.8,-1.2),(-2.2,-.8),(-2.2,.5),(-2.1,.5)])):
 for x,y in pts: goto(i,x,y)
 act((0,0,CAL) if i==0 else (0,0,NOOP),(0,0,CAL) if i==1 else (0,0,NOOP),f'cal{i}')
 act((0,0,IMAGE) if i==0 else (0,0,NOOP),(0,0,IMAGE) if i==1 else (0,0,NOOP),f'image{i}')
 act((0,0,SEND) if i==0 else (0,0,NOOP),(0,0,SEND) if i==1 else (0,0,NOOP),f'send{i}')

# return home via side waypoints
for i,pts in enumerate(([(2.2,-.7),(2.0,-1.4),(1,-1.75)], [(-2.1,-.5),(-1.9,-1.45),(-1,-1.45),(-1,-1.75)])):
 for x,y in pts: goto(i,x,y)
 act((0,0,SEND) if i==0 else (0,0,NOOP),(0,0,SEND) if i==1 else (0,0,NOOP),f'home-send{i}')
print('FIRST_TERM',firstterm)
if mode=='nodrop':
 act((0,0,DROP),(0,0,DROP),'final-drop')
 print('FIRST_TERM_AFTER_DROP',firstterm)
env.close()
