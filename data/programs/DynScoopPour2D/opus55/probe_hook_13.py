from probe_hook_lib import *
def trial(seed,arm,depth,rx_max=3.27):
    p=P(seed); h=p.h(); bx=h['x']-0.025; top=h['y']+0.5
    # gripper point G = bar-center at y = top - depth(ish); robot at distance (arm+0.02+... ) along -dir
    rx=min(bx,rx_max); dxg=bx-rx; L=arm+0.07   # 0.07: mid-finger point
    th=-np.pi/2+np.arcsin(dxg/L) if dxg<L else None
    gy=top-0.05
    ry=gy+L*np.cos(np.arcsin(dxg/L))
    p.goto(th=th); p.goto(y=max(1.4,ry+0.3)); p.goto(x=rx,arm=arm); p.goto(y=ry)
    h0=p.h(); p.step([0,0,0,0,-0.015],12); h1=p.h(); r=p.r()
    p.goto(y=1.3); p.goto(x=2.8); h2=p.h()
    print(seed,'arm',arm,'robot',r,'held',h1['held'],'shift',round(h1['x']-h0['x'],3),round(h1['theta']-h0['theta'],3),'after carry',h2); p.env.close()
trial(0,0.2,0)
trial(1,0.4,0); trial(1,0.3,0)
for s in [2,3,4,5]: trial(s,0.3,0)
