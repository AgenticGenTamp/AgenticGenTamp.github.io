import sys; sys.path.insert(0,"scratch"); from h import *
for base in [(-0.6,0,0),(-0.6,-0.3,0),(-0.55,-0.45,0)]:
    o,info=env.reset(seed=1); o,r0=goto(o,(base[0],0,0),rstate(o)[3:10]); o,r0b=goto(o,base,rstate(o)[3:10])
    b=blocks(o)[1]; p=bpose(o,b); y=byaw(o,b); y=y-np.round(y/(np.pi/2))*np.pi/2
    o,r1=move_tool(o,base,[p[0],p[1],0.95],y)
    zs=None
    for z in np.arange(0.94,0.79,-0.01):
        o,r=move_tool(o,base,[p[0],p[1],z],y,steps=1)
        if r: zs=z; break
    print(base,r0,r0b,r1,"rej at",zs,tool(o)[0].round(3))
