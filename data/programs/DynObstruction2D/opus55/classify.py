import sys
from env_client import make_env
env=make_env()
for s in map(int, sys.argv[1:]):
    obs,info=env.reset(seed=s)
    g=lambda n,f: obs.get(obs.get_object_from_name(n),f)
    bx,bw,sx,sw=g('target_block','x'),g('target_block','width'),g('target_surface','x'),g('target_surface','width')
    gap = bx-bw/2 if sx>bx else 3.236-(bx+bw/2)
    obst=[(round(obs.get(o,'x'),2),round(obs.get(o,'width'),2),round(obs.get(o,'height'),2)) for o in obs.data if o.name.startswith('obs')]
    print(s,'blk x%.2f w%.2f h%.2f'%(bx,bw,g('target_block','height')),'surf x%.2f w%.2f'%(sx,sw),'wallgap %.2f'%gap, obst)
