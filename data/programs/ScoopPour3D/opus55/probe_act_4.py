exec(open('probe_act_1.py').read().split('def run')[0])
z=[0.]*11
def go(seq,label,j):
    obs,_=env.reset(seed=0); s0=st(obs); out=[]
    for a in seq:
        obs,*_=env.step(np.array(a,float)); out.append(round(float(st(obs)[j]-s0[j]),4))
    print(label, out)
def A(j,v): a=list(z); a[j]=v; return a
go([A(3,0.1)]+[z]*30,"j1 +0.1 once, 30 zeros",3)
go([A(3,-0.1)]*20+[z]*20,"j1 -0.1 x20, then 20 zeros",3)
go([A(3,0.05)]*20+[z]*10,"j1 +0.05 x20",3)
go([A(4,0.1)]*20+[z]*20,"j2 +0.1 x20",4)
go([A(7,0.1)]*10+[z]*10,"j5 +0.1 x10",7)
go([A(6,0.1)]*10+[z]*10,"j4 +0.1 x10",6)
