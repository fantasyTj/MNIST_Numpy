import numpy as np
from dataset.mnist import load_mnist
from deepcovnet import DeepConvNet
from components.trainer import Trainer

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,flatten=False,one_hot_label=True)

network=DeepConvNet()
trainer=Trainer(network,x_train,t_train,x_test,t_test,20,100,feedback=True)
trainer.train()

network.save_params("deepconvnet_params.pkl")
print("saved!")
