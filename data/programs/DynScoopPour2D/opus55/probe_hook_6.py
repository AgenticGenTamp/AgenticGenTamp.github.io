from probe_hook_lib import *
def probe(start, d, th=np.pi, thr=3e-4):
    p=P(); p.goto(th=th); p.goto(y=0.8); p.goto(x=start[0]); p.goto(y=start[1]); p.goto(x=start[0],y=start[1])
    h0=p.h(); p.step([0]*5,5); h1=p.h(); cause=None
    for k in range(1500):
        r0=p.r(); p.step([d[0]*0.0005,d[1]*0.0005,0,0,0]); h=p.h(); r=p.r()
        if r['x']==r0['x'] and r['y']==r0['y']: cause='blocked';break
        if abs(h['x']-h1['x'])+abs(h['y']-h1['y'])+abs(h['theta']-h1['theta'])>thr: cause='hookmoved';break
    print('start',start,'dir',d,cause,'robot',r['x'],r['y'],'hook',h,'rest-jitter',h0,h1)
    p.env.close()
probe((2.6,0.35),(1,0))
probe((3.35,0.3),(-1,0),th=0)
