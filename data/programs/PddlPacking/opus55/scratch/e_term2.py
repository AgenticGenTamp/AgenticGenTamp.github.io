import sys; sys.path.insert(0,"scratch"); from h import *
o,info=env.reset(seed=1); o,_=goto(o,BASE,rstate(o)[3:10])
b0,b1=blocks(o)
o,ok,_,r=pick(o,b0); o,rej=carry(o,b0,(-0.03,0.06),0.0); o=grip(o,1)
for y in [-0.12,-0.095,-0.09,-0.087,-0.085,-0.08]:
    o,ok,_,r=pick(o,b1)
    if not ok: print("pick fail",r); break
    o,rej=carry(o,b1,(-0.03,y),np.pi/4); o=grip(o,1)
    print("b1 y",y,"rej",rej,bpose(o,b1)[[0,1,2,7]].round(4),round(byaw(o,b1),3),"te",LAST['te'])
    if LAST['te']: break
