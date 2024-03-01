import numpy as np

def softmax(inp):
    if inp.ndim==1:
        inp=inp-np.max(inp)
        return np.exp(inp)/np.sum(np.exp(inp))

    row_max=np.max(inp,axis=1).reshape(inp.shape[0],-1)
    inp=inp-row_max
    return np.exp(inp)/(np.sum(np.exp(inp),axis=1).reshape(inp.shape[0],-1))

def cross_entropy_error(sm,t):
    if sm.ndim==1:
        sm=sm.reshape(1,sm.size)
        t=t.reshape(1,t.size)

    # 将one_hot数据转换
    if sm.size==t.size:
        t=np.argmax(t,axis=1)

    batch_size=sm.shape[0]
    return -np.sum(np.log(sm[np.arange(batch_size),t]+1e-7))/batch_size

def im2col(inp,fh,fw,stride,pad):
    n,c,w,h=inp.shape
    out_h = 1 + (h + 2 * pad - fh) // stride
    out_w = 1 + (w + 2 * pad - fw) // stride

    img=np.pad(inp,[(0,0),(0,0),(pad,pad),(pad,pad)],'constant')
    col=np.zeros((n,c,out_h,out_w,fh,fw))

    for i in range(out_h):
        for j in range(out_w):
            col[:,:,i,j,:,:]=img[:,:,i*stride:i*stride+fh,j*stride:j*stride+fw]

    col=col.transpose(0,2,3,1,4,5).reshape(n*out_h*out_w,-1)
    return col

def col2im(col,img_shape,fh,fw,stride,pad):
    n,c,h,w=img_shape
    out_h=1+(h+2*pad-fh)//stride
    out_w=1+(w+2*pad-fw)//stride

    col=col.reshape(n,out_h,out_w,c,fh,fw).transpose(0,3,1,2,4,5)

    img=np.zeros((n,c,h+2*pad,w+2*pad))

    for i in range(out_h):
        for j in range(out_w):
            img[:,:,i*stride:i*stride+fh,j*stride:j*stride+fw]+=col[:,:,i,j,:,:]

    return img[:,:,pad:h+pad,pad:w+pad]



