try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np


url_base='http://yann.lecun.com/exdb/mnist/'
key_file={'train_img':'train-images-idx3-ubyte.gz',
          'train_label':'train-labels-idx1-ubyte.gz',
          'test_img':'t10k-images-idx3-ubyte.gz',
          'test_label':'t10k-labels-idx1-ubyte.gz' }

dataset_dir=os.path.dirname(os.path.abspath(__file__))
save_file=dataset_dir+"/mnist.pkl"

train_num=60000
test_num=10000
img_dim=(1,28,28)
img_size=784

def _download(file_name):
    file_path=dataset_dir+"/"+file_name
    if os.path.exists(file_path):
        return

    print("Downloading "+file_name+"...")
    urllib.request.urlretrieve(url_base+file_name,file_path)  #从网址获取,下载成压缩包
    print("Done")

def download_mnist():
    for v in key_file.values():
        _download(v)

def _load_label(file_name):
    file_path=dataset_dir+"/"+file_name

    print("Converting "+file_name+"to NumPy Array...")
    with gzip.open(file_path,'rb') as f:
        labels=np.frombuffer(f.read(),np.uint8,offset=8)  #label和image文件用的格式是IDX 每个文件前面会有作者自己定义的魔数(magic number)用来做文件完整性校验 所以要offset跳过这个魔数
    print("Done")

    return labels

def _load_img(file_name):
    file_path=dataset_dir+"/"+file_name

    print("Converting "+file_name+"to NumPy Array...")
    with gzip.open(file_path,'rb') as f:
        data=np.frombuffer(f.read(),np.uint8,offset=16)  #label和image文件用的格式是IDX 每个文件前面会有作者自己定义的魔数(magic number)用来做文件完整性校验 所以要offset跳过这个魔数
    data=data.reshape(-1,img_size) #-1代表行数依其他条件决定，如列数为784，那么行数row*784就应该等于原来的矩阵
    print("Done")

    return data

def _convert_numpy():
    dataset={}
    dataset['train_img']=_load_img((key_file['train_img']))
    dataset['train_label']=_load_label(key_file['train_label'])
    dataset['test_img']=_load_img(key_file['test_img'])
    dataset['test_label']=_load_label(key_file['test_label'])

    return dataset

def init_mnist():
    download_mnist()  #先下载到四个压缩包中
    dataset=_convert_numpy()  #从压缩包中写到dataset
    print("Creating pickle file...")
    with open(save_file,'wb') as f:  #从dateset写到pickle
        pickle.dump(dataset,f,-1)  #当第三个参数的值是负数，使用最高的协议对dataset压缩
    print("Done")

def _change_one_hot_label(X):
    T=np.zeros((X.size,10))
    for idx,row in enumerate(T):  #返回元素(row)及其索引(idx)的一个元组
        row[X[idx]]=1
    return T

def load_mnist(normalize=True,flatten=True,one_hot_label=False):

    if not os.path.exists(save_file):  # 查看是否已下载到pickle中
        init_mnist()

    with open(save_file,'rb') as f:
        dataset=pickle.load(f)  # 从f中读取一个字符串，并将它重构为原来的python对象

    if normalize:   # normalize : 将图像的像素值正规化为0.0~1.0
        for key in ('train_img','test_img'):
            dataset[key]=dataset[key].astype(np.float32)
            dataset[key]/=255.0

    if one_hot_label:
        dataset['train_label']=_change_one_hot_label(dataset['train_label'])
        dataset['test_label']=_change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img','test_img'):
            dataset[key]=dataset[key].reshape(-1,1,28,28)  #-1代表依据实际情况决定

    return (dataset['train_img'],dataset['train_label']),(dataset['test_img'],dataset['test_label'])

if __name__=='__main__':
    init_mnist()







