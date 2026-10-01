from env_client import make_env
import numpy as np
env = make_env()
bad=0;badg=0;gaps=[]
for seed in range(300):
    obs, info = env.reset(seed=seed)
    g=lambda n,f: obs.get(obs.get_object_from_name(n),f)
    bx,bw,sx=g('target_block','x'),g('target_block','width'),g('target_surface','x')
    if sx>bx: gap=bx-bw/2
    else: gap=3.236-(bx+bw/2)
    if gap<0.49:
        bad+=1; badg+= bw<0.25; gaps.append(round(gap,2))
print(bad,badg,sorted(gaps))
