from geo_lib import *
import sys
seed=int(sys.argv[1]); c=(float(sys.argv[2]),float(sys.argv[3]))
P=Prober(seed,'part1')
Pobj=P.obs.get_object_from_name('part1')
print('type',P.obs.get(Pobj,'triangle_type'),'quat',np.round(P.quat,4))
print('center ok',floor_ok(P,*c))
r={}
r['xmax']=bisect(P,lambda t:(t,c[1]),c[0],0.45)[0]
r['xmin']=bisect(P,lambda t:(t,c[1]),c[0],0.15)[0]
r['ymax']=bisect(P,lambda t:(c[0],t),c[1],0.2)[0]
r['ymin']=bisect(P,lambda t:(c[0],t),c[1],-0.2)[0]
print({k:round(v,4) for k,v in r.items()})
# extents from pose: wall inner x 0.21..0.39, y -0.14..0.14
print('ext +x',round(0.39-r['xmax'],4),'-x',round(r['xmin']-0.21,4),'+y',round(0.14-r['ymax'],4),'-y',round(r['ymin']+0.14,4))
# corner tests: pose so bbox overshoots corner by 0.015 in both axes
ex={'+x':0.39-r['xmax'],'-x':r['xmin']-0.21,'+y':0.14-r['ymax'],'-y':r['ymin']+0.14}
for sx in (1,-1):
  for sy in (1,-1):
    x=(0.39-ex['+x']+0.015) if sx>0 else (0.21+ex['-x']-0.015)
    y=(0.14-ex['+y']+0.015) if sy>0 else (-0.14+ex['-y']-0.015)
    print('corner',sx,sy,(round(x,4),round(y,4)),floor_ok(P,x,y))
print('steps',P.nsteps)
