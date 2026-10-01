from pngdec import load
w,h,bpp,rows=load('mcp_renders/state_seed3.png')
def lab(r,g,b):
    if r>150 and g>150 and b<100:return 'Y'
    if r>100 and g<80 and b<80:return 'R'
    if g>100 and r<80 and b<80:return 'G'
    if b>100 and r<80 and g<80:return 'B'
    return None
import collections
for c,(xa,xb) in {'B':(200,300),'G':(200,300),'Y':(340,440),'R':(340,440)}.items():
    ys=collections.defaultdict(list)
    for y in range(290,420):
        for x in range(xa,xb):
            if lab(*rows[y][x*bpp:x*bpp+3])==c: ys[y].append(x)
    yy=sorted(ys); 
    if c in 'BY': yy=[y for y in yy if y<345]
    else: yy=[y for y in yy if y>340]
    print(c,'rows',yy[0],yy[-1])
    for y in yy[::4]: print('  ',y,min(ys[y]),max(ys[y]),len(ys[y]))
