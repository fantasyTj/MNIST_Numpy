import numpy as np
from final_project.components.optimizer import *

class Trainer:
    def __init__(self,network,x_train,t_train,x_test,t_test,epochs,batch_size,feedback=True):
        self.network=network
        self.feedback=feedback
        self.x_train=x_train
        self.t_train=t_train
        self.x_test=x_test
        self.t_test=t_test
        self.epochs=epochs
        self.batch_size=batch_size


        # optimizer
        self.optimizer=Adam()

        self.train_size=x_train.shape[0]
        self.iter_num=self.train_size//self.batch_size
        self.total_iter_num=self.epochs*self.iter_num
        self.cur_iter=0
        self.cur_epoch=0

        self.train_loss_list=[]
        self.train_acc_list=[]
        self.test_acc_list=[]

    def train_step(self):
        batch_mask=np.random.choice(self.train_size,self.batch_size)
        batch_x=self.x_train[batch_mask]
        batch_t=self.t_train[batch_mask]

        grads=self.network.gradient(batch_x,batch_t)
        self.optimizer.update(self.network.params,grads)

        loss=self.network.loss(batch_x,batch_t)
        self.train_loss_list.append(loss)

        if self.feedback:
            print("train loss:"+str(loss))

        if self.cur_iter%self.iter_num==0:
            self.cur_epoch+=1

            random_test_mask=np.random.choice(self.x_test.shape[0],1000)
            x_test_sample,t_test_sample=self.x_test[random_test_mask],self.t_test[random_test_mask]

            test_acc=self.network.accuracy(x_test_sample,t_test_sample)
            self.test_acc_list.append(test_acc)

            if self.feedback:
                print("epoch "+str(self.cur_epoch)+' test_acc: '+str(test_acc))

        self.cur_iter+=1

    def train(self):
        for i in range(self.total_iter_num):
            self.train_step()

        final_test_acc=self.network.accuracy(self.x_test,self.t_test)

        if self.feedback:
            print("The final test acc is: "+str(final_test_acc))
