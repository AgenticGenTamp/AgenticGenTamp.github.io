import sys; sys.path.insert(0,"scratch"); from e_z import *
for z in [0.845,0.85,0.855]: print(z,trial(z))
# table collision height over empty table / off table
for xy in [(-0.2,-0.1),(-0.45,-0.1)]:
    o=setup(); o,rej=move_tool(o,BASE,list(xy)+[0.9],0.0)
    for z in np.arange(0.9,0.6,-0.004):
        o,rej=move_tool(o,BASE,list(xy)+[z],0.0,steps=1)
        if rej: print(xy,"rej at z",round(z,3),"tool at",tool(o)[0].round(4)); break
# block: descend with fine steps over block1
o=setup(); b=blocks(o)[1]; p=bpose(o,b); yaw=byaw(o,b)
o,_=move_tool(o,BASE,[p[0],p[1],0.9],yaw)
for z in np.arange(0.9,0.6,-0.002):
    o,rej=move_tool(o,BASE,[p[0],p[1],z],yaw,steps=1)
    if rej: print("block rej at z",round(z,3),"tool",tool(o)[0].round(4)); break
# block with gripper rotated 45
o=setup(); o,_=move_tool(o,BASE,[p[0],p[1],0.9],yaw+np.pi/4)
for z in np.arange(0.9,0.6,-0.002):
    o,rej=move_tool(o,BASE,[p[0],p[1],z],yaw+np.pi/4,steps=1)
    if rej: print("block45 rej at z",round(z,3),"tool",tool(o)[0].round(4)); break
