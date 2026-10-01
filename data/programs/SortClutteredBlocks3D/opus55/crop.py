import zlib,struct
from pngdec import load
w,h,bpp,rows=load('mcp_renders/state_seed3.png')
x0,y0,x1,y1,S=280,335,360,395,8
out=b''
for y in range(y0,y1):
    line=b''
    for x in range(x0,x1):
        line+=bytes(rows[y][x*bpp:x*bpp+3])*S
    out+=(b'\x00'+line)*S
W=(x1-x0)*S;H=(y1-y0)*S
def chunk(t,d):return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
open('crop3.png','wb').write(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',W,H,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(out))+chunk(b'IEND',b''))
