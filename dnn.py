import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)

import torch.utils.data as Data
from sklearn.model_selection import train_test_split


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
    
        self.data_input_numpy = data_input # 生成输入数据

        # 生成输出数据 
        data_output = data_odom_extratct[2:data_size+2 ,[0,1,2] ].astype(np.float)

        # 放大部分输出数据（扩大部分数据作用）
        data_output[:, 2] = data_output[:, 2] *10 # 转角

        self.data_output_numpy = data_output #生成输出数据

        # ------------1.3 产生训练集和测试集----------

        data_train, data_test, label_train, label_test = self.data_loader()

        # ------------1.3 将numpy转化成tensor----------

        #训练集
        data_train_torch = torch.from_numpy(data_train)
        self.data_train_torch = data_train_torch.float() # 转化成浮点数(重点语句不要错过）

        data_label_torch = torch.from_numpy(label_train)
        self.data_label_torch = data_label_torch.float() # 转化成浮点数(重点语句不要错过）

        #测试集
        data_test_torch = torch.from_numpy(data_test)
        self.data_test_torch = data_test_torch.float() # 转化成浮点数(重点语句不要错过）

        label_test_torch = torch.from_numpy(label_test)
        self.label_test_torch = label_test_torch.float() # 转化成浮点数(重点语句不要错过）


    #----------------2.构建神经网络----------------
    def dnn(self):

        # ------------2.1进行批训练-----------

        # ------------2.1.1载入训练集-----------

        # 定义数据库 （输入输出分别是之前的输入输出）
        dataset_train = Data.TensorDataset(self.data_train_torch, self.data_label_torch) 

        # 定义数据加载器
        loader_train = Data.DataLoader(dataset = dataset_train, batch_size = self.BATCH_SIZE, shuffle = True, num_workers = 2)

        # ------------2.1.2载入测试集-----------

        # 定义数据库 （输入输出分别是之前的输入输出）
        dataset_test = Data.TensorDataset(self.data_test_torch, self.label_test_torch) 

        # 定义数据加载器
        loader_test = Data.DataLoader(dataset = dataset_test, batch_size = self.BATCH_SIZE, shuffle = True, num_workers = 2)


        #------------2.2初始化部分数据-----------
        #定义迭代次数 
        times = self.data_size

        # 绘图：生成随机输出变量
        self.error_train = torch.zeros(times,3) #设定了一千条数据
        self.error_test = torch.zeros(times,3) #设定了一千条数据

        # 绘图：生成损失函数误差变量
        self.loss_train = torch.zeros(times,1) #设定了一千条数据
        self.loss_test = torch.zeros(times,1) #设定了一千条数据

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
        
        # 训练集
        for epoch in range(5):
            for step, (batch_x, batch_y) in enumerate(loader_train):
                
                #print("batch_x.cuda()")
                #print(batch_x.cuda())
                
                #产生prediction
                prediction = net( batch_x.cuda() )     # input x and predict based on x

                #计算loss
                loss = loss_func(prediction, batch_y.cuda())     # must be (1. nn output, 2. target)

                print ("loss")
                print (loss)

                #将误差函数画出来
                self.loss_train[step,:] = loss 

                #计算optimize
                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients

                #计算误差百分数,并储存在data_plot当中
                #percent_1 = 100*(prediction - 10*batch_y.cuda())/10*batch_y.cuda()
                percent = prediction - batch_y.cuda() #直接计算误差大小

                #print("error")
                #print(percent)

                self.error_train[step,:] = percent[0,:]  #设定了一千条数据
               

        # 测试集
        for epoch in range(1):
            for step, (batch_x, batch_y) in enumerate(loader_test):
                #产生prediction
                prediction = net( batch_x.cuda() )     # input x and predict based on x

                #计算loss
                loss = loss_func(prediction, batch_y.cuda())     # must be (1. nn output, 2. target)

                #print ("loss_测试")
                #print (loss)

                #将误差函数画出来
                self.loss_test[step,:] = loss

                #直接计算误差大小
                percent = prediction - batch_y.cuda() 

                #print("error")
                #print(percent)

                self.error_test[step,:] = percent[0,:]

        #将训练完的网络保存
        torch.save(net.state_dict(), 'params.pkl')
        
       

    #----------------3.将函数loss画出来----------------
    def plot_loss(self):

        #将数据从tensor转化成numpy
        loss_train_numpy = self.loss_train.detach().numpy() #self 
        loss_test_numpy = self.loss_test.detach().numpy() #self 

        #将误差平均值可视化
        X = np.linspace(1, self.data_size, self.data_size, endpoint=True) #self

        #设置XY轴的显示范围
        #plt.xlim(0, self.data_size/self.BATCH_SIZE) #可能需要人为调整
        #plt.ylim(0, 50) #可能需要人为调整

        #绘图:两个子图 
        plt.figure(1)

        plt.subplot(211)
        plt.xlabel('Time(s)')
        plt.ylabel("loss of train")
        plt.xlim(0, (0.9*self.data_size)/self.BATCH_SIZE) #可能需要人为调整
        plt.plot(X,loss_train_numpy)

        plt.subplot(212)
        plt.xlabel('Time(s)')
        plt.ylabel("loss of test")
        plt.xlim(0, (0.1*self.data_size)/self.BATCH_SIZE) #可能需要人为调整
        plt.plot(X,loss_test_numpy)

        plt.show()

    #----------------3.将函数loss画出来----------------
    def plot_error(self):

        #----------------3.1训练集&测试集----------------
        #将数据从tensor转化成numpy
        error_train_numpy = self.error_train.detach().numpy() #self 
        #取每个元素的绝对值
        error_train_numpy_abs = np.abs(error_train_numpy.T) #需要进行转置才能得到相关数据
        #调用sum函数，将每列数据加起来，求误差的平均数
        average_train = np.sum(error_train_numpy_abs, axis=0)/3 #要除3，表示加起来求平均

        #将数据从tensor转化成numpy
        error_test_numpy = self.error_test.detach().numpy() #self 
        #取每个元素的绝对值
        error_test_numpy_abs = np.abs(error_test_numpy.T) #需要进行转置才能得到相关数据
        #调用sum函数，将每列数据加起来，求误差的平均数
        average_test = np.sum(error_test_numpy_abs, axis=0)/3 #要除3，表示加起来求平均

        #----------------3.2 画图----------------
        #将误差平均值可视化
        X = np.linspace(1, self.data_size, self.data_size, endpoint=True) #self

        #设置XY轴的显示范围
        plt.xlim(0, self.data_size/self.BATCH_SIZE) #可能需要人为调整
        #plt.ylim(0, 50) #可能需要人为调整

        #绘图
        plt.figure(1)

        plt.subplot(211)
        plt.xlabel('Time(s)')
        plt.ylabel("error of train")
        plt.xlim(0, (0.9*self.data_size)/self.BATCH_SIZE) #可能需要人为调整
        plt.plot(X,average_train)

        plt.subplot(212)
        plt.xlabel('Time(s)')
        plt.ylabel("error of test")
        plt.xlim(0, (0.1*self.data_size)/self.BATCH_SIZE) #可能需要人为调整
        plt.plot(X,average_test)
        
        plt.show()
     

    def data_loader(self):
        #产生训练集和测试集
        data_train, data_test, label_train, label_test = train_test_split(self.data_input_numpy, self.data_output_numpy, test_size=0.1, random_state=42)

        #将训练集和测试集返回
        return data_train, data_test, label_train, label_test

    def restore_params(self):
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

        #显示网络
        print("net_restore")
        print(net)
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