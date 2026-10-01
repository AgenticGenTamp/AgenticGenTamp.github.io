from plib import *
env=make_env()
for s in range(40):
    obs,_=env.reset(seed=s)
    objs=[(n,d(obs,n)) for n in obs.get_object_names() if n!='robot' and n!='target_surface']
    for i in range(len(objs)):
        for j in range(len(objs)):
            if i==j: continue
            a,b=objs[i][1],objs[j][1]
            gap=b['x']-(a['x']+a['width'])
            dt=abs((a['y']+a['height'])-(b['y']+b['height']))
            if -0.001<gap<0.03 and dt<0.013:
                print(f"seed {s}: {objs[i][0]}(x{a['x']:.3f},w{a['width']:.3f},top{a['y']+a['height']:.4f}) | {objs[j][0]}(x{b['x']:.3f},top{b['y']+b['height']:.4f}) gap={gap:.4f} dtop={dt:.4f} n={len(objs)}")
env.close()
