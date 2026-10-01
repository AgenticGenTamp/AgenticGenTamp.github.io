import numpy as np, sys
from pngread import read_png

def classify(px):
    r,g,b = int(px[0]),int(px[1]),int(px[2])
    mx,mn = max(r,g,b), min(r,g,b)
    if mx-mn < 40: return None
    if r>100 and g>100 and b<100: return 'yellow'
    if r>mx*0.8 and g<mx*0.6 and b<mx*0.6: return 'red'
    if g>=mx*0.8 and r<mx*0.6 and b<mx*0.6: return 'green'
    if b>=mx*0.8 and r<mx*0.6 and g<mx*0.6: return 'blue'
    return None

def blobs(path):
    im = read_png(path)
    h,w,_ = im.shape
    lab = np.full((h,w), '', dtype=object)
    for y in range(h):
        for x in range(w):
            c = classify(im[y,x])
            if c: lab[y,x]=c
    # connected components
    seen = np.zeros((h,w),bool); out=[]
    for y in range(h):
        for x in range(w):
            if lab[y,x] and not seen[y,x]:
                c = lab[y,x]; st=[(y,x)]; seen[y,x]=True; pts=[]
                while st:
                    cy,cx = st.pop(); pts.append((cy,cx))
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx = cy+dy,cx+dx
                        if 0<=ny<h and 0<=nx<w and not seen[ny,nx] and lab[ny,nx]==c:
                            seen[ny,nx]=True; st.append((ny,nx))
                if len(pts)>=6:
                    ys=[p[0] for p in pts]; xs=[p[1] for p in pts]
                    out.append((c,len(pts),float(np.mean(ys)),float(np.mean(xs))))
    return out
if __name__=='__main__':
    for b in sorted(blobs(sys.argv[1]), key=lambda t:t[1]):
        print(b)
