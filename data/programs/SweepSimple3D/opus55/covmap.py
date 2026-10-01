import json,numpy as np,glob,sys
A=np.concatenate([np.array(json.load(open(f)),float) for f in glob.glob(sys.argv[1] if len(sys.argv)>1 else 'logs/[A-H]_s1.json')])
A=A[(A[:,6]>0.013)&(A[:,6]<0.1)]
xs=np.arange(-3.0,2.8,0.2); ys=np.arange(2.6,-3.4,-0.2)
print('      '+''.join('%-5.1f'%x if i%5==0 else '' for i,x in enumerate(xs)))
for y in ys:
    row=''
    for x in xs:
        m=(A[:,4]>=x)&(A[:,4]<x+0.2)&(A[:,5]>=y)&(A[:,5]<y+0.2)
        row+='#' if m.sum()>=2 else ('+' if m.sum() else '.')
    print('%5.1f '%y+ ''.join(c for c in row))
