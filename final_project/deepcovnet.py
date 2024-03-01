import pickle
import numpy as np
from components.layers import *

class DeepConvNet:
    def __init__(self,input_shape=(1,28,28),
                 conv_params_1={'filter_num':16,'filter_size':3,'pad':1,'stride':1},
                 conv_params_2={'filter_num':16,'filter_size':3,'pad':1,'stride':1},
                 conv_params_3={'filter_num':32,'filter_size':3,'pad':1,'stride':1},
                 conv_params_4={'filter_num':32,'filter_size':3,'pad':2,'stride':1},
                 conv_params_5={'filter_num':64,'filter_size':3,'pad':1,'stride':1},
                 conv_params_6={'filter_num':64,'filter_size':3,'pad':1,'stride':1},
                 hidden_size=50,output_size=10):
        each_output_shape=np.zeros((9,3))
        each_output_shape[0]=input_shape
        each_output_shape[7]=(1,1,hidden_size)
        each_output_shape[8]=(1,1,output_size)

        conv_params_lst=[conv_params_1,conv_params_2,conv_params_3,conv_params_4,conv_params_5,conv_params_6]
        for i in range(1,7):
            each_output_shape[i][0]=conv_params_lst[i-1]['filter_num']
            out_h=(each_output_shape[i-1][1]+2*conv_params_lst[i-1]['pad']-conv_params_lst[i-1]['filter_size'])//conv_params_lst[i-1]['stride']+1
            each_output_shape[i][1]=out_h
            each_output_shape[i][2]=out_h

            if i%2==0:
                each_output_shape[i][1]/=2
                each_output_shape[i][2]/=2 # 池化层处理

        pre_node_nums=np.zeros((8,))
        for i in range(6):
            pre_node_nums[i]=each_output_shape[i][0]*(conv_params_lst[i]['filter_size'])**2
        pre_node_nums[6]=each_output_shape[6][0]*(each_output_shape[6][1])**2
        pre_node_nums[7]=hidden_size

        w_init=np.sqrt(2.0/pre_node_nums) # He初始值

        self.params={}
        pre_c_num=input_shape[0]
        for idx,conv_param in enumerate(conv_params_lst):
            self.params['w'+str(idx+1)]=w_init[idx]*np.random.randn(conv_param['filter_num'],pre_c_num,conv_param['filter_size'],conv_param['filter_size'])
            self.params['b'+str(idx+1)]=np.zeros(conv_param['filter_num'])
            pre_c_num=conv_param['filter_num']
        self.params['w7']=w_init[6]*np.random.randn(int(pre_node_nums[6]),hidden_size)
        self.params['b7']=np.zeros(hidden_size)
        self.params['w8']=w_init[7]*np.random.randn(hidden_size,output_size)
        self.params['b8']=np.zeros(output_size)

        self.layers=[]
        self.layers.append(Convolution(self.params['w1'],self.params['b1'],conv_params_1['stride'],conv_params_1['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['w2'],self.params['b2'],conv_params_2['stride'],conv_params_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2,pool_w=2,stride=2))

        self.layers.append(Convolution(self.params['w3'],self.params['b3'],conv_params_3['stride'],conv_params_3['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['w4'],self.params['b4'],conv_params_4['stride'],conv_params_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2,pool_w=2,stride=2))

        self.layers.append(Convolution(self.params['w5'], self.params['b5'], conv_params_5['stride'], conv_params_5['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['w6'], self.params['b6'], conv_params_6['stride'], conv_params_6['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        self.layers.append(Affine(self.params['w7'],self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['w8'],self.params['b8']))
        self.layers.append(Dropout(0.5))

        self.last_layer=SoftmaxWithLoss()

    def predict(self,inp,is_training=False):
        for layer in self.layers:
            if isinstance(layer,Dropout):
                inp=layer.forward(inp,is_training)
            else:
                inp=layer.forward(inp)

        # batch_size*10的二维矩阵
        return inp

    def loss(self,inp,t):
        inp2=self.predict(inp,is_training=True)
        return self.last_layer.forward(inp2,t)

    def accuracy(self,inp,t):
        if t.ndim!=1:
            t=np.argmax(t,axis=1)

        count=0
        batch_size=100

        for i in range(inp.shape[0]//batch_size):
            batch_inp=inp[i*batch_size:(i+1)*batch_size]
            batch_t=t[i*batch_size:(i+1)*batch_size]
            res=np.argmax(self.predict(batch_inp,is_training=False),axis=1)
            count+=np.sum(res==batch_t)

        return count/inp.shape[0]

    def gradient(self,inp,t):
        # forward
        self.loss(inp,t)

        # backward
        d_out=1
        d_out=self.last_layer.backward()

        reversed_layers=self.layers.copy()
        reversed_layers.reverse()
        for layer in reversed_layers:
            d_out=layer.backward(d_out)

        grads={}
        for i,layer_idx in enumerate((0,2,5,7,10,12,15,18)):
            grads['w'+str(i+1)]=self.layers[layer_idx].d_w
            grads['b'+str(i+1)]=self.layers[layer_idx].d_b

        return grads

    def save_params(self,file_name='params.pkl'):
        params={}
        for key,val in self.params.items():
            params[key]=val

        with open(file_name,'wb') as f:
            pickle.dump(params,f)

    def load_params(self,file_name='params.pkl'):
        with open(file_name,'rb') as f:
            params=pickle.load(f)

        for key,val in params.items():
            self.params[key]=val

        for i,layer_idx in enumerate((0,2,5,7,10,12,15,18)):
            self.layers[layer_idx].w=self.params['w'+str(i+1)]
            self.layers[layer_idx].b=self.params['b'+str(i+1)]
