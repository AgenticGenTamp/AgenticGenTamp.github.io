import zlib, struct, numpy as np
def readpng(path):
    d=open(path,'rb').read(); assert d[:8]==b'\x89PNG\r\n\x1a\n'
    i=8; idat=b''; pal=None; trns=None
    while i<len(d):
        ln=struct.unpack('>I',d[i:i+4])[0]; typ=d[i+4:i+8]; data=d[i+8:i+8+ln]; i+=12+ln
        if typ==b'IHDR': w,h,bd,ct,comp,filt,inter=struct.unpack('>IIBBBBB',data)
        elif typ==b'IDAT': idat+=data
        elif typ==b'PLTE': pal=np.frombuffer(data,np.uint8).reshape(-1,3)
        elif typ==b'IEND': break
    assert bd==8 and inter==0, (bd,inter)
    ch={0:1,2:3,3:1,4:2,6:4}[ct]
    raw=zlib.decompress(idat)
    stride=w*ch
    out=np.zeros((h,stride),np.uint8)
    prev=np.zeros(stride,np.uint8); p=0
    for y in range(h):
        f=raw[p]; p+=1
        line=np.frombuffer(raw[p:p+stride],np.uint8).astype(np.int32).copy(); p+=stride
        pr=prev.astype(np.int32)
        if f==0: cur=line
        elif f==1:
            cur=line
            for x in range(ch,stride): cur[x]=(cur[x]+cur[x-ch])&255
        elif f==2: cur=(line+pr)&255
        elif f==3:
            cur=line
            for x in range(stride):
                a=cur[x-ch] if x>=ch else 0
                cur[x]=(cur[x]+((a+pr[x])>>1))&255
        elif f==4:
            cur=line
            for x in range(stride):
                a=cur[x-ch] if x>=ch else 0
                b=pr[x]; c=pr[x-ch] if x>=ch else 0
                pp=a+b-c; pa=abs(pp-a); pb=abs(pp-b); pc=abs(pp-c)
                pred=a if (pa<=pb and pa<=pc) else (b if pb<=pc else c)
                cur[x]=(cur[x]+pred)&255
        cur=cur.astype(np.uint8)
        out[y]=cur; prev=cur
    img=out.reshape(h,stride//ch if ch else w,ch) if ct!=3 else None
    if ct==3:
        idx=out.reshape(h,w); img=pal[idx]
    elif ct==0: img=np.repeat(out.reshape(h,w,1),3,2)
    elif ct==4: img=np.repeat(out.reshape(h,w,2)[:,:,:1],3,2)
    else: img=out.reshape(h,w,ch)[:,:,:3]
    return img.astype(np.uint8)
