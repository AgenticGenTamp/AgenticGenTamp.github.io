import sys; sys.path.insert(0,"scratch"); from h import *
B2=(-0.45,-0.2,0.0)
def setupAB(seed=0):
    o,info=env.reset(seed=seed); o,_=goto(o,BASE,rstate(o)[3:10])
    B=blocks(o); A=B[1]; Bb=B[0]
    o,ok,_,r=pick(o,A); assert ok,("pickA",r)
    o,rej=carry(o,A,(-0.05,0.0),0.0)
    o=grip(o,1); print("A placed",bpose(o,A).round(3),round(byaw(o,A),3))
    o,r=goto(o,B2,rstate(o)[3:10]); print("base move rej",r)
    o,ok,_,r=pick(o,Bb,base=B2); assert ok,("pickB",r)
    return o,A,Bb
def sweep(o,A,Bb,byw,ds,dirn=(1,0)):
    for d in ds:
        tgt=(-0.05+d*dirn[0],d*dirn[1])
        o,rej=carry(o,Bb,tgt,byw,base=B2)
        o=grip(o,1); s=rstate(o)
        print("yaw",round(byw,3),"d",d,"rej",rej,"B",bpose(o,Bb)[[0,1,2,7]].round(4),round(byaw(o,Bb),3),"robot ga",s[11],"gopen",round(s[10],3),"te",LAST['te'])
        if s[11]<0.5: return o,d
    return o,None
if __name__=="__main__":
    o,A,Bb=setupAB()
    o,d=sweep(o,A,Bb,0.0,[0.05,0.06,0.069,0.0695,0.07,0.0705,0.072,0.08])
