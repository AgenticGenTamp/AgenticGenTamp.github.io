from probe_hook_lib import *
def probe(start, d, th=np.pi):
    p=P(); p.goto(th=th); p.goto(y=0.8); p.goto(x=start[0]); p.goto(y=start[1]); p.goto(x=start[0],y=start[1])
    h0=p.h(); cause=None
    for k in range(1500):
        r0=p.r(); p.step([d[0]*0.001,d[1]*0.001,0,0,0]); h=p.h(); r=p.r()
        if r['x']==r0['x'] and r['y']==r0['y']: cause='blocked';break
        if abs(h['x']-h0['x'])+abs(h['y']-h0['y'])>1e-3: cause='hookmoved';break
    print('start',start,'dir',d,cause,'robot',r['x'],r['y'],'hook',h)
    p.env.close()
probe((2.3,0.3),(1,0))
probe((2.3,0.25),(1,0))
probe((2.3,0.22),(1,0))
probe((2.85,0.8),(0,-1))
