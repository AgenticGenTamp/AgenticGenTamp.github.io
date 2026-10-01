import sys; sys.path.insert(0,"scratch"); from h import *
o,info=env.reset(seed=0); o,_=goto(o,BASE,rstate(o)[3:10])
b=blocks(o)[0]; p=bpose(o,b); yaw=byaw(o,b)
for dy in [0,np.pi/2]:
    o,info=env.reset(seed=0); o,_=goto(o,BASE,rstate(o)[3:10])
    o,r=move_tool(o,BASE,[p[0],p[1],0.95],yaw+dy); print(dy,"above",r,tool(o)[0].round(3),rstate(o)[3:10].round(2))
    for z in np.arange(0.94,0.79,-0.01):
        o,r=move_tool(o,BASE,[p[0],p[1],z],yaw+dy,steps=1)
        if r: print(" rej at",z,tool(o)[0].round(3)); break
