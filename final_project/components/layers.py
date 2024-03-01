import sys,os
sys.path.append(os.pardir)
import numpy as np
from final_project.components.functions import softmax,cross_entropy_error,im2col,col2im

class Relu:
    def __init__(self):
        self.mask=None

    def forward(self,inp):
        self.mask=(inp<=0)
        out=inp.copy()
        out[self.mask]=0

        return out

    def backward(self,d_out):
        d_out[self.mask]=0

        return d_out

class Affine:
    def __init__(self,w,b):
        self.w=w
        self.b=b

        self.d_w=None
        self.d_b=None

        self.x=None
        self.x_shape=None

    def forward(self,inp):
        self.x_shape=inp.shape
        self.x=inp.reshape(inp.shape[0],-1)

        out=np.dot(self.x,self.w)+self.b

        return out

    def backward(self,d_out):
        d_x=np.dot(d_out,self.w.T)
        self.d_w=np.dot(self.x.T,d_out)
        self.d_b=np.sum(d_out,axis=0)

        d_x=d_x.reshape(*self.x_shape)

        return d_x

class SoftmaxWithLoss:
    def __init__(self):
        self.loss=None
        self.sm=None
        self.t=None

    def forward(self,inp,t):
        self.t=t
        self.sm=softmax(inp)
        self.loss=cross_entropy_error(self.sm,self.t)

        return self.loss

    def backward(self):
        # 默认是one_hot数据
        d_x=(self.sm-self.t)/self.t.shape[0]

        return d_x

class Dropout:
    def __init__(self,dropout_ratio=0.5):
        self.dropout_ratio=dropout_ratio
        self.mask=None

    def forward(self,inp,is_training=True):
        if is_training:
            self.mask=(np.random.rand(*inp.shape)>self.dropout_ratio)
            return inp*self.mask
        else:
            return inp*(1-self.dropout_ratio)

    def backward(self,d_out):
        return d_out*self.mask

class Convolution:
    def __init__(self,w,b,stride=1,pad=0):
        self.w=w
        self.b=b
        self.stride=stride
        self.pad=pad

        self.d_w=None
        self.d_b=None

        # backward时使用的中间数据
        self.inp=None
        self.col_inp=None
        self.col_w=None

    def forward(self,inp):
        fn,c,fh,fw=self.w.shape # filter
        n,c,h,w=inp.shape

        out_h=1+(h+2*self.pad-fh)//self.stride
        out_w=1+(w+2*self.pad-fw)//self.stride

        col_inp=im2col(inp,fh,fw,self.stride,self.pad)
        col_w=self.w.reshape(fn,-1).T

        out=np.dot(col_inp,col_w)+self.b
        out=out.reshape(n,out_h,out_w,-1).transpose(0,3,1,2)

        self.inp=inp
        self.col_inp=col_inp
        self.col_w=col_w

        return out

    def backward(self,d_out):
        fn,c,fh,fw=self.w.shape
        d_out=d_out.transpose(0,2,3,1).reshape(-1,fn)

        self.d_b=np.sum(d_out,axis=0)
        self.d_w=np.dot(self.col_inp.T,d_out)
        self.d_w=self.d_w.transpose(1,0).reshape((fn,c,fh,fw))

        d_col_inp=np.dot(d_out,self.col_w.T)
        d_x=col2im(d_col_inp,self.inp.shape,fh,fw,self.stride,self.pad)

        return d_x

class Pooling:
    def __init__(self,pool_h,pool_w,stride,pad=0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.stride=stride
        self.pad=pad

        self.inp=None
        self.max_mask=None

    def forward(self,inp):
        n,c,h,w=inp.shape
        out_h=int(1+(h-self.pool_h)/self.stride)
        out_w=int(1+(w-self.pool_w)/self.stride)

        col=im2col(inp,self.pool_h,self.pool_w,self.stride,self.pad)
        col=col.reshape(-1,self.pool_h*self.pool_w)

        max_mask=np.argmax(col,axis=1)
        out=np.max(col,axis=1)
        out=out.reshape((n,out_h,out_w,c)).transpose(0,3,1,2)

        self.inp=inp
        self.max_mask=max_mask

        return out

    def backward(self,d_out):
        d_out=d_out.transpose(0,2,3,1)

        pool_size=self.pool_h*self.pool_w
        d_max=np.zeros((d_out.size,pool_size))
        d_max[np.arange(self.max_mask.size),self.max_mask]=d_out.flatten()
        d_max=d_max.reshape(d_out.shape+(pool_size,))

        d_col=d_max.reshape(d_max.shape[0]*d_max.shape[1]*d_max.shape[2],-1)
        d_x=col2im(d_col,self.inp.shape,self.pool_h,self.pool_w,self.stride,self.pad)

        return d_x




