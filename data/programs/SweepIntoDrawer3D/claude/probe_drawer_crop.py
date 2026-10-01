import sys, zlib, struct, numpy as np
from pngtool import readpng

def writepng(path, arr):
    h,w,_ = arr.shape
    raw = b''.join(b'\x00'+arr[y].tobytes() for y in range(h))
    def chunk(t,d):
        c = t+d
        return struct.pack('>I',len(d))+c+struct.pack('>I',zlib.crc32(c)&0xffffffff)
    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB',w,h,8,2,0,0,0))
    png += chunk(b'IDAT', zlib.compress(raw,6))
    png += chunk(b'IEND', b'')
    open(path,'wb').write(png)

def crop(src, out, x0,y0,x1,y1, k=4, grid=0):
    a = readpng(src)[y0:y1, x0:x1]
    b = np.repeat(np.repeat(a,k,0),k,1)
    if grid:
        for i,x in enumerate(range(x0,x1)):
            if x % grid == 0: b[:, (x-x0)*k, :] = np.array([255,0,255])
        for y in range(y0,y1):
            if y % grid == 0: b[(y-y0)*k, :, :] = np.array([0,255,255])
    writepng(out, b.astype(np.uint8))
    print(out, a.shape, '->', b.shape, 'origin', (x0,y0), 'scale', k)

if __name__=='__main__':
    s=sys.argv
    crop(s[1], s[2], int(s[3]),int(s[4]),int(s[5]),int(s[6]), int(s[7]) if len(s)>7 else 4, int(s[8]) if len(s)>8 else 0)
