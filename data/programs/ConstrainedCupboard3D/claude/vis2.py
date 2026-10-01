import numpy as np, sys
from env_client import make_env
cnt=int(sys.argv[1]); zs=[float(x) for x in sys.argv[2].split(',')]; ys=[float(x) for x in sys.argv[3].split(',')]
env=make_env(); o,info=env.reset(seed=0,options={'object_count':cnt})
qw,qz=np.cos(np.pi/4),np.sin(np.pi/4)
names=sorted([n for n in o.get_object_names() if n.startswith('cuboid')])
for i,n in enumerate(names):
    d=o.data[o.get_object_from_name(n)]
    d[0],d[1],d[2]=2.0,ys[i%len(ys)],zs[i%len(zs)]; d[3],d[4],d[5],d[6]=qw,0,0,qz
print(env.render_state(state=o,label='c%d'%cnt))
env.close()
