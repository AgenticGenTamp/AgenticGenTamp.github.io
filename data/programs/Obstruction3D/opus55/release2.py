from th import *
h=H(1); p=h.pose('target_block'); he=h.he('target_block'); top=p[2]+he[2]
h.goto([p[0],p[1],top+0.1],0)
h.goto([p[0],p[1],top-0.01],0); h.grip(-1); print('grasp',h.grasped())
tgt=[p[0]+0.08,p[1]-0.15]
ok,_=h.goto([tgt[0],tgt[1],top+0.1],0); print('moved',ok, h.pose('target_block')[:3].round(4))
z=top+0.1
while z>0.0:
    z-=0.005
    ok,e=h.goto([tgt[0],tgt[1],z],0)
    if not ok:
        print('blocked at tool z',z,'block',h.pose('target_block')[:3].round(4), 'tool', h.tool()[:3,3].round(4)); break
for i in range(2):
    h.grip(1); print('open', h.grasped(), h.pose('target_block')[:3].round(4))
h.goto([tgt[0],tgt[1],top+0.1],0); print('after lift, block', h.pose('target_block')[:3].round(4), h.grasped())
