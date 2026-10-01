import zlib, struct
import numpy as np
def write_png(path, img):
    h,w,_=img.shape
    raw=b''.join(b'\x00'+img[y].astype(np.uint8).tobytes() for y in range(h))
    def chunk(t,d):
        c=struct.pack('>I',len(d))+t+d
        return c+struct.pack('>I', zlib.crc32(t+d)&0xffffffff)
    png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
    open(path,'wb').write(png)
def crop_zoom(src,dst,r0,r1,c0,c1,k=4):
    from pngread import read_png
    im=read_png(src)[r0:r1,c0:c1]
    im=np.repeat(np.repeat(im,k,axis=0),k,axis=1)
    write_png(dst,im)
