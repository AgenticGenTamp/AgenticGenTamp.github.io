import math
from env_client import make_env

e=make_env();matches=0;extremes=[]
for seed in range(400,1501):
 s,info=e.reset(seed=seed)
 sh=next(iter(s.get_objects(e.observation_space.get_type('shelf'))))
 sx,sy,sw=[float(s.get(sh,f)) for f in ['x1','y1','width1']]
 cols=max(1,int(round(sw/.3175)))
 for b in s.get_objects(e.observation_space.get_type('target_block')):
  x,y,t,w,h=[float(s.get(b,f)) for f in ['x','y','theta','width','height']]
  bx=x+math.cos(t)*w/2-math.sin(t)*h/2;by=y+math.sin(t)*w/2+math.cos(t)*h/2
  if by<sy:continue
  slot=sx+(int((bx-sx)/(sw/cols))+.5)*sw/cols
  lateral=.075 if slot<.25 else (-.075 if slot>4.75 else 0.)
  if not lateral:continue
  a=t+math.pi/2
  basex=bx-(.2+h/2+.028)*math.cos(a)+lateral*math.sin(a)
  if basex<.201 or basex>4.799:
   print('MATCH',seed,info,b.name,'center',bx,by,'theta',t,'slot',slot,'base',basex,'shelf',(sx,sy,sw),flush=True);matches+=1
  if matches>=2:break
 if matches>=2:break
 if seed%100==0:print('PROGRESS',seed,flush=True)
print('DONE matches',matches,'lastseed',seed,flush=True)
e.close()
