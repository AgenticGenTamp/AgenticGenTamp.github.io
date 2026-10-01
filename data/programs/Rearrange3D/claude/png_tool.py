import zlib, struct, numpy as np

def read_png(p):
    d=open(p,'rb').read(); assert d[:8]==b'\x89PNG\r\n\x1a\n'
    i=8; idat=b''; w=h=bd=ct=None
    while i<len(d):
        ln=struct.unpack('>I',d[i:i+4])[0]; typ=d[i+4:i+8]; data=d[i+8:i+8+ln]; i+=12+ln
        if typ==b'IHDR': w,h,bd,ct,_,_,il=struct.unpack('>IIBBBBB',data); assert il==0
        elif typ==b'IDAT': idat+=data
        elif typ==b'IEND': break
    ch={0:1,2:3,3:1,4:2,6:4}[ct]; assert bd==8
    raw=zlib.decompress(idat); stride=w*ch
    out=np.zeros((h,stride),dtype=np.uint8); pos=0; prev=np.zeros(stride,dtype=np.uint8)
    for y in range(h):
        f=raw[pos]; pos+=1
        line=np.frombuffer(raw[pos:pos+stride],dtype=np.uint8).copy(); pos+=stride
        if f==0: cur=line
        elif f==1:
            cur=line
            for x in range(ch,stride): cur[x]=(int(cur[x])+int(cur[x-ch]))&255
        elif f==2: cur=(line.astype(int)+prev.astype(int)).astype(np.uint8)
        elif f==3:
            cur=line
            for x in range(stride):
                a=int(cur[x-ch]) if x>=ch else 0
                cur[x]=(int(cur[x])+((a+int(prev[x]))>>1))&255
        elif f==4:
            cur=line
            for x in range(stride):
                a=int(cur[x-ch]) if x>=ch else 0
                c=int(prev[x-ch]) if x>=ch else 0
                b=int(prev[x]); pp=a+b-c
                pa,pb,pc=abs(pp-a),abs(pp-b),abs(pp-c)
                pr=a if (pa<=pb and pa<=pc) else (b if pb<=pc else c)
                cur[x]=(int(cur[x])+pr)&255
        out[y]=cur; prev=cur
    return out.reshape(h,w,ch)[:,:,:3]

def write_png(p,arr):
    h,w,_=arr.shape; raw=b''.join(b'\x00'+arr[y].tobytes() for y in range(h))
    def chunk(t,d): 
        c=struct.pack('>I',len(d))+t+d; return c+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
    open(p,'wb').write(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b''))
