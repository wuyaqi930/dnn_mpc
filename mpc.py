#-------------导入相关安装包-------------
import torch
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import reload_dnn as reload #载入训练网络的package 
import time #导入时间
import datetime

# 定义一个模型预测控制的类 
class MPC:

    #------------1.定义数据传入函数------------
    def __init__(self,x,x_d,prediction_horizon,state_number,input_number):

        self.x_d=x_d #将理想轨迹传入

        print("self.x_d __init__")
        print(self.x_d)

        self.x=x #将实际轨迹传入
        #self.x[0,:] = self.x_d[10,:] #第一行数据给了一个估计数值

        self.prediction_horizon=prediction_horizon #预测范围

        self.state_number=state_number #状态量

        self.input_number=input_number #输入量

    #------------初始化相关数值------------

        self.Q=1000*np.eye(prediction_horizon) #初始化Q（需要半正定）

        self.R=10*np.eye(prediction_horizon) #初始化R（需要正定）

        self.u = np.random.rand(prediction_horizon,input_number) #初始化实际状态量（列向量）为随机数

        #调试代码
        #print("x_d")
        #print(self.x_d)

        #print("x")
        #print(self.x)

        #print("prediction_horizon")
        #print(self.prediction_horizon)

        #print("state_number")
        #print(self.state_number)

        #print("input_number")
        #print(self.input_number)

        #print("Q")
        #print(self.Q)

        #print("R")
        #print(self.R)

        #print("u")
        #print(self.u)
    #------------初始化网络------------
        net = reload.reload_dnn() #初始化

        self.reload = net.load_dnn() #重新载入网络

        #调试代码
        #print("reload mpc") #print出来看看
        #print(self.reload)

        #solution = minimize(self.objective,self.x,args=(self.x_d,self.u,self.Q,self.R),method='SLSQP',constraints=cons) #求解过程放在初始化里面就可以

    #------------定义目标代价函数------------
    #def objective(self,*args):
    def objective(self,*args):
        #args的unpack
        x,x_d,u,Q,R = args

        #将x来reshape成（30,3）矩阵:原因是x被初始化成了（90*1）
        x=np.reshape(x,(self.prediction_horizon,self.state_number) )

        #计算代价函数--第一部分
    
        #计算所有项的误差的平方
        error_x = np.power(x-x_d,2) 

        #将每一项加起来
        sum_x=np.sum(error_x)
       
        #计算代价函数总的部分
        J = sum_x 

        #print("J")
        #print(J)

        return J 
    

    #------------定义运动学函数------------
    def f(self,x,u):

        f=np.zeros(3) #输入输出是一维的变量，直接定义成数组就行，不要定义成矩阵

        f[0]= x[0]+ np.cos(u[0])*0.01 # 坐标x
        f[1]= x[1]+ np.sin(u[0])*0.01 # 坐标y
        f[2]= x[2]+ np.sin(u[1])*0.01 # 转角theta

        return f

    ##------------定义运动学函数：用网络进行运动学估计------------
    #def f(self,x,u):
    #    #f = 2*x  # 实际函数可能需要改
    #    print("f(self,x,u) x")
    #    print(x)

    #    print("f(self,x,u) u")
    #    print(u)

    #    #将X和u合并产生
    #    input= np.hstack((x,u))
    #    input_torch = torch.from_numpy(input)
        
    #    print("f(self,x,u) input")
    #    print(input_torch.float())

    #    #调用网络进行运动学估计
    #    output = self.reload(input_torch.float())

    #    print("f(self,x,u) output")
    #    print(output)

    #    return output.detach().numpy() 

    # 1.输入变量符合运动学方程 （等式）
    #def constraint1(self,*args):
    def constraint1(self,*args):
        #数据unpack
        x,u,num = args
        
        #将x来reshape成（10,3）矩阵
        x=np.reshape(x,(self.prediction_horizon,self.state_number) )

        #对u的数据处理
        return self.f(x[num,:],u) - x[num+1,:]


    # 2.初始状态达到理想数值(等式）
    def constraint2(self,*args):
        #args的unpack
        x,x_d=args

        #将x来reshape成（10,3）矩阵
        x=np.reshape(x,(self.prediction_horizon,self.state_number) )   

        #对x_d的处理
        x_d = np.array(x_d,dtype=float) # 将元组转化为数组
        x_d = np.reshape(x_d,(self.prediction_horizon,self.state_number) )  # 将数组reshape

        #取数据的size
        lenth = len(x[:,0]) # 将prediction horizon取出来

        return x[0,:] - x_d[0,:] # 初始数值要为零


    # 3.最终状态达到理想数值(等式）
    def constraint3(self,*args):
        #args的unpack
        x,x_d=args

        #将x来reshape成（10,3）矩阵
        x=np.reshape(x,(self.prediction_horizon,self.state_number) )
        
        #对x_d的处理
        x_d = np.array(x_d,dtype=float) # 将元组转化为数组
        x_d = np.reshape(x_d,(self.prediction_horizon,self.state_number) )  # 将数组reshape

        #取数据的size
        lenth = len(x[:,0]) # 将prediction horizon取出来
       
        return x[lenth-1,:] - x_d[lenth-1,:] # 最终的数值要为零

    #------------开始进行优化------------
    
    def optimize(self):
        # 定义空的list 
        cons= []

        # 1.定义运动学约束
        for num in range(self.prediction_horizon-1):
            con = {'type': 'eq', 'fun': self.constraint1,'args':(self.u[num,:],num)}
            cons.append (con)
        
        # 2.定义初始状态约束
        con10 = {'type': 'eq', 'fun': self.constraint2,'args':(self.x_d,)} 
        #cons.append(con10)

        # 3.定义最终状态约束
        con11 = {'type': 'eq', 'fun': self.constraint3,'args':(self.x_d,)} 
        cons.append(con11)

        # 总约束：cons
        #localtime_start = time.time()
        starttime = datetime.datetime.now()

        # 求解:1.传入的优化数据是self.x 2.传入的其他数据是args 3.优化数据会传入constrains当中（非常重要）4.传入的参数都被优化了，没传入的参数没有被优化（除了x状态量）
        #solution = minimize(self.objective,self.x,args=(self.x_d,self.u,self.Q,self.R),method='SLSQP',constraints=cons) # 会把初始状态导入数据
        solution = minimize(self.objective,self.x,args=(self.x_d,self.u,self.Q,self.R),method='SLSQP',constraints=cons) # 会把初始状态导入数据
        x = solution.x 
   
        #localtime_end = time.time()
        endtime = datetime.datetime.now()

        print("开始时间")
        print(starttime)

        print("结束时间")
        print(endtime)

        print("运行时间")
        print(endtime-starttime)

        #将x来reshape成（10,3）矩阵
        x=np.reshape(x,(self.prediction_horizon,self.state_number) )

        print("x 最终")
        print(x)

        print("self.u")
        print(self.u)

        return x,self.u #将控制数据和状态估计返回



   


