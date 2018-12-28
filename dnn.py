import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)

import torch.utils.data as Data


class neural_network:

    #------------1.初始化------------
    def __init__(self,data_odom,data_size,BATCH_SIZE): 

        #初始化赋值
        self.data_odom = data_odom #里程计数值
        self.data_size = data_size #数据量大小
        self.BATCH_SIZE = BATCH_SIZE #训练批处理size

        #------------1.1 数据筛选------------

        #针对非平面问题的数据筛选
        data_odom_extratct =  self.data_odom[ : , [0,5,6,7,8,9,10,11,48,49,50,51,52,53]]

        # 针对平面问题的数据筛选
        #只筛选特定的行
        data_odom_extratct = data_odom_extratct[ : , [1,2,6,8,13]]

        #------------1.2 数据拼接&生成输入输出------------

        # 取数据前2000行
        data_odom_extratct_data_size = data_odom_extratct[ 1:self.data_size+1 , :]

        #生成输入数据 
        data_input = data_odom_extratct_data_size.astype(np.float)

        #放大部分输入数据（扩大部分数据作用）
        data_input[:, 2] = data_input[:, 2] *10 # 转角
        data_input[:, 4] = data_input[:, 4] *10 # 角速度
    
        data_input = data_input # 生成输入数据

        # 生成输出数据 
        data_output = data_odom_extratct[2:data_size+2 ,[0,1,2] ].astype(np.float)

        # 放大部分输出数据（扩大部分数据作用）
        data_output[:, 2] = data_output[:, 2] *10 # 转角

        data_output = data_output #生成输出数据

        # ------------1.3 将numpy转化成tensor-----------

        data_input_torch = torch.from_numpy(data_input)
        self.data_input_float = data_input_torch.float() # 转化成浮点数(重点语句不要错过）

        data_output_torch = torch.from_numpy(data_output)
        self.data_output_float = data_output_torch.float() # 转化成浮点数(重点语句不要错过）


    #----------------2.构建神经网络----------------
    def dnn(self):

        # ------------2.1进行批训练-----------
        # 定义数据库 （输入输出分别是之前的输入输出）
        dataset = Data.TensorDataset(self.data_input_float, self.data_output_float) 

        # 定义数据加载器
        loader = Data.DataLoader(dataset = dataset, batch_size = self.BATCH_SIZE, shuffle = True, num_workers = 2)

        #------------2.2初始化部分数据-----------
        #定义迭代次数 
        times = self.data_size

        # 绘图：生成随机输出变量
        self.data_plot = torch.zeros(times,3) #设定了一千条数据

        # 绘图：生成损失函数误差变量
        self.loss_plot = torch.zeros(times,3) #设定了一千条数据

        #----------------2.3构建网络-----------------
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

        net = Net(n_feature=5, n_hidden=64, n_output=3) 

        print(net)

        #----------------2.4定义优化方法&定义损失函数-----------------

        #使用“随机梯度下降法”进行参数优化
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)  # 传入 net 的所有参数, 学习率

        #使用“ADAM”进行参数优化
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0005) # 传入 net 的所有参数, 学习率

        #定义损失函数，计算均方差
        #loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)
        loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

        #----------------2.5使用cuda进行GPU计算-----------------
        net.cuda()
        loss_func.cuda()

        #----------------2.6具体训练过程-----------------
        for epoch in range(5):
            for step, (batch_x, batch_y) in enumerate(loader):
                #产生prediction
                prediction = net( batch_x.cuda() )     # input x and predict based on x

                #计算loss
                loss = loss_func(prediction, batch_y.cuda())     # must be (1. nn output, 2. target)

                print ("loss")
                print (loss)

                #计算optimize
                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients

                #计算误差百分数,并储存在data_plot当中
                percent = 100*(prediction - batch_y.cuda())/batch_y.cuda()

                self.data_plot[step,:] = percent[0,:] # 取precent矩阵的第一行 (这边有个self)

    #----------------3.将函数误差画出来----------------
    def plot(self):
        #将数据从tensor转化成numpy
        data_plot_numpy = self.data_plot.detach().numpy() #self 

        #取每个元素的绝对值
        data_plot_numpy_abs = np.abs(data_plot_numpy.T) #需要进行转置才能得到相关数据
        
        #调用sum函数，将每列数据加起来，求误差的平均数
        average = np.sum(data_plot_numpy_abs, axis=0)/3 #要除3，表示加起来求平均

        #将误差平均值可视化
        X = np.linspace(1, self.data_size, self.data_size, endpoint=True) #self

        #设置XY轴的显示范围
        plt.xlim(7000, 7500) #可能需要人为调整
        plt.ylim(0, 50) #可能需要人为调整

        #绘图
        plt.plot(X,average)
        plt.show()