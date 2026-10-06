exec(open('dbg2.py').read().split('rng=np.random')[0])
for y in [0.11,0.12,0.13,0.15,0.18]:
  row=''
  for x in np.arange(0.3,1.3,0.05):
    best=0
    for th in np.linspace(-np.pi,np.pi,16,endpoint=False):
      best|=chk0.free(np.array([x,y,th,0.1]))
    row+= '1' if best else '.'
  print(y,row)
print(rects['obstruction13']); 
for n,d in rects.items():
    c=rect_verts(d['x'],d['y'],d['th'],d['w'],d['h']).mean(0)
    if np.linalg.norm(c-q0[:2])<0.6: print(n,np.round(c,2),round(d['w'],2),round(d['h'],2),round(d['th'],2))
