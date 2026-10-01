import zlib,struct,sys
def load(fn):
    d=open(fn,'rb').read();i=8;idat=b''
    while i<len(d):
        l,=struct.unpack('>I',d[i:i+4]);t=d[i+4:i+8];c=d[i+8:i+8+l];i+=12+l
        if t==b'IHDR':w,h,bd,ct=struct.unpack('>IIBB',c[:10])
        if t==b'IDAT':idat+=c
    raw=zlib.decompress(idat);bpp={2:3,6:4}[ct];st=w*bpp;rows=[];prev=bytearray(st);p=0
    for y in range(h):
        f=raw[p];line=bytearray(raw[p+1:p+1+st]);p+=1+st
        for x in range(st):
            a=line[x-bpp] if x>=bpp else 0;b=prev[x];c=prev[x-bpp] if x>=bpp else 0
            if f==1:line[x]=(line[x]+a)&255
            elif f==2:line[x]=(line[x]+b)&255
            elif f==3:line[x]=(line[x]+(a+b)//2)&255
            elif f==4:
                pa=abs(b-c);pb=abs(a-c);pc=abs(a+b-2*c)
                pr=a if pa<=pb and pa<=pc else (b if pb<=pc else c);line[x]=(line[x]+pr)&255
        rows.append(line);prev=line
    return w,h,bpp,rows
def lab(r,g,b):
    if r>200 and g>200 and b>200:return '.'
    if r>150 and g>150 and b<100:return 'Y'
    if r>100 and g<80 and b<80:return 'R'
    if g>100 and r<80 and b<80:return 'G'
    if b>100 and r<80 and g<80:return 'B'
    if r<60 and g<60 and b<60:return '#'
    return ' '
