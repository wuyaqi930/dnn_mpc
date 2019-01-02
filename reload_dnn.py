import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)

import torch.utils.data as Data
from sklearn.model_selection import train_test_split



#-------------将训练好的DNN重新载入-------------
class reload_dnn:

    #初始化
    def __init__(self):
        self.dnn = None

    #定义网络
    def load_dnn(self):
        #定义网络类型
        class Net(torch.nn.Module):
            def __init__(self, n_feature, n_hidden, n_output):
                super(Net,self).__init__()
                self.hidden = torch.nn.Sequential(
                    torch.nn.Linear(n_feature, n_hidden),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(n_hidden, n_hidden),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(n_hidden, n_hidden),
                    torch.nn.LeakyReLU()
                )
                self.out = torch.nn.Linear(n_hidden, n_output)
            def forward(self, x):
                x=self.hidden(x)
                out =  self.out(x)
                return out
          
        #新建net
        net = Net(n_feature=5, n_hidden=64, n_output=3) 

        #载入参数
        net.load_state_dict(torch.load('params.pkl'))

        ##显示网络
        #print("net_restore")
        #print(net)
        ## ------------测试是否成功重新载入:通过计算误差来看实际效果-----------

        ##新建net
        #net = Net(n_feature=5, n_hidden=64, n_output=3) 
        #net.cuda() #cuda加速

        ##新建loss
        #loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)
        #loss_func.cuda()
        
        ##载入参数
        #net.load_state_dict(torch.load('params.pkl'))

        #print("net_restore")
        #print(net)

        ## 定义数据库 （输入输出分别是之前的输入输出）
        #dataset_restore = Data.TensorDataset(self.data_train_torch, self.data_label_torch) 

        ## 定义数据加载器
        #loader_restore = Data.DataLoader(dataset = dataset_restore, batch_size = self.BATCH_SIZE, shuffle = True, num_workers = 2)

        ## 定义误差矩阵
        ##定义迭代次数 
        #times = self.data_size

        ## 绘图：生成随机输出变量
        #self.error_restore = torch.zeros(times,3) #设定了一千条数据
        #self.loss_restore = torch.zeros(times,3) #设定了一千条数据


        ## ------------实际计算-----------
        #for epoch in range(1):
        #    for step, (batch_x, batch_y) in enumerate(loader_restore):
        #        #产生prediction
        #        prediction = net( batch_x.cuda() )     # input x and predict based on x

        #        #计算loss
        #        loss_restore = loss_func(prediction, batch_y.cuda())     # must be (1. nn output, 2. target)


        #        print ("loss_restore")
        #        print (loss_restore)

        #        #将误差函数画出来
        #        self.loss_restore[step,:] = loss_restore

        #        #直接计算误差大小
        #        percent_restore = prediction - batch_y.cuda() 

        #        print("error")
        #        print(percent_restore)

        #        self.error_restore[step,:] = percent_restore[0,:]

        ## ------------画图-----------

        ## ------------画loss函数------------
        ##将数据从tensor转化成numpy
        #loss_restore_numpy = self.loss_restore.detach().numpy() #self 

        ##将误差平均值可视化
        #X = np.linspace(1, self.data_size, self.data_size, endpoint=True) #self

        ##设置XY轴的显示范围
        #plt.xlim(0, (0.9*self.data_size)/self.BATCH_SIZE) #可能需要人为调整

        ##绘图
        #plt.plot(X,loss_restore_numpy)
        #plt.show()

        ## ------------画error函数------------
        ##将数据从tensor转化成numpy
        #error_restore_numpy = self.error_restore.detach().numpy() #self 

        ##取每个元素的绝对值
        #error_restore_numpy_abs = np.abs(error_restore_numpy.T) #需要进行转置才能得到相关数据

        ##调用sum函数，将每列数据加起来，求误差的平均数
        #average_restore = np.sum(error_restore_numpy_abs, axis=0)/3 #要除3，表示加起来求平均

        ##将误差平均值可视化
        #X = np.linspace(1, self.data_size, self.data_size, endpoint=True) #self

        ##设置XY轴的显示范围
        #plt.xlim(0, (0.9*self.data_size)/self.BATCH_SIZE) #可能需要人为调整
        ##plt.ylim(0, 50) #可能需要人为调整

        ##绘图
        #plt.plot(X,average_restore)
        #plt.show()
        return net