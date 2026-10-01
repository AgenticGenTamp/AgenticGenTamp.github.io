import sys; sys.path.insert(0,"scratch"); from h import *
B3=(-0.45,0.0,0.0)
o,info=env.reset(seed=0); o,_=goto(o,BASE,rstate(o)[3:10]); b=blocks(o)[1]
o,ok,y,r=pick(o,b); o,r=goto(o,B3,rstate(o)[3:10]); print("base move",r)
for z in [1.1,1.2]:
    o,rej=carry(o,b,(0.0,0.0),0.0,z=z,base=B3); print("carry z",z,"rej",rej,"tool",tool(o)[0].round(3),"block",bpose(o,b)[:3].round(3))
o=grip(o,1); print("open high: block",bpose(o,b)[[0,1,2,7]].round(4),"gopen",rstate(o)[10])
o,ok,y,r=pick(o,b,base=B3); print("repick",ok)
o,rej=carry(o,b,(-0.36,0.1),0.0,base=B3); print("carry off-table rej",rej,bpose(o,b)[:3].round(3))
o=grip(o,1); print("open off table: block",bpose(o,b)[[0,1,2,7]].round(4),"robot ga",rstate(o)[11])
