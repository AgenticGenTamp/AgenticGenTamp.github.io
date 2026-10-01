import sys; sys.path.insert(0,"scratch"); from h import *
YS=[-0.13,-0.134,-0.136,-0.1695,-0.17,-0.1,-0.099,-0.098,-0.097,-0.096,-0.095]
o,info=env.reset(seed=1); print(info); o,_=goto(o,BASE,rstate(o)[3:10])
b0,b1=blocks(o)
o,ok,_,r=pick(o,b0); print("pick0",ok,r)
o,rej=carry(o,b0,(-0.03,0.06),0.0); o=grip(o,1); print("b0",bpose(o,b0)[[0,1,2,7]].round(3),"te",LAST['te'],LAST['r'])
for y in YS:
    o,ok,_,r=pick(o,b1); 
    if not ok: print("pick fail",r); break
    o,rej=carry(o,b1,(-0.03,y),0.0); o=grip(o,1)
    print("b1 y",y,"rej",rej,bpose(o,b1)[[0,1,2,7]].round(4),round(byaw(o,b1),3),"te",LAST['te'],"r",LAST['r'],LAST['info'])
    if LAST['te']: break
