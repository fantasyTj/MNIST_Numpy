import numpy as np

class Adam:
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.count=0
        self.m=None
        self.v=None

    def update(self,params,grads):
        if self.m is None:
            self.m,self.v={},{}
            for key,val in params.items():
                self.m[key]=np.zeros_like(val)
                self.v[key]=np.zeros_like(val)

        self.count+=1
        lr_=self.lr*np.sqrt(1.0-self.beta2**self.count)

        for key in params.keys():
            self.m[key]+=(1.0-self.beta1)*(grads[key]-self.m[key])
            self.v[key]+=(1.0-self.beta2)*(grads[key]**2-self.v[key])

            params[key]-=lr_*self.m[key]/(np.sqrt(self.v[key])+1e-7)