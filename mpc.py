#coding=utf-8
#-------------导入相关安装包-------------
#import torch
import numpy as np
from scipy.optimize import minimize
#import matplotlib.pyplot as plt
#import reload_dnn as reload #载入训练网络的package 
import time #导入时间
import datetime

# 定义一个模型预测控制的类 
class MPC:

    #------------1.定义数据传入函数------------
    def __init__(self,x,x_init,x_d,prediction_horizon,state_number,input_number):
        self.x_d=x_d #将理想轨迹传入

        self.x=x #将实际轨迹传入

        self.x_init=x_init #将初始轨迹传入

        self.u_init=np.zeros(2) #将控制量初始值初始化

        self.prediction_horizon=prediction_horizon #预测范围

        self.state_number=state_number #状态量

        self.input_number=input_number #输入量

    #------------初始化相关数值------------

        self.Q=1000 #初始化Q（需要半正定）

        self.R=10 #初始化R（需要正定）

        self.u = np.ones((prediction_horizon,input_number)) #初始化实际状态量（列向量）为随机数
        #self.u = np.zeros((prediction_horizon,input_number)) #初始化实际状态量（列向量）为随机数

        # 优化量= 状态量+控制量：将两个矩阵合并
        self.x_plus_u = np.append(self.x,self.u.astype(int),axis=1)

        # 优化量初始值 = 状态量初始值+控制量初始值：将两个矩阵合并
        self.x_plus_u_init = np.append(self.x_init,self.u_init.astype(int))

        # 最终状态的误差数值
        self.final_error = 0.2


    #------------初始化网络------------  (调试：暂时不初始化)
        # net = reload.reload_dnn() #初始化
        # self.reload = net.load_dnn() #重新载入网络

    #------------定义目标代价函数------------
    def objective(self,*args):
        #args的unpack
        x_plus_u,x_d,Q,R = args

        # 调试代码
        # print("x_d")
        # print(x_d)

        # print("Q")
        # print(Q)

        # print("R")
        # print(R)

        #将x来reshape成（10,5）矩阵:原因是x被初始化成了（50*1）
        x_plus_u=np.reshape(x_plus_u,(self.prediction_horizon,self.state_number+self.input_number) )

        # 调试代码
        # print("x")
        # print(x_plus_u[:,0:3])

        # print("u")
        # print(x_plus_u[:,3:5])
        
        #计算代价函数--第一部分
    
        #计算所有项的误差的平方
        error_x = self.Q*np.power(x_plus_u[:,0:3]-x_d,2) # x = x_plus_u[:,0:3], self.Q:比例系数（公式中：Q）

        # 调试代码
        # print("error_x")
        # print(error_x)

        #将每一项加起来
        sum_x=np.sum(error_x)

        # 调试代码
        # print("sum_x")
        # print(sum_x)
        
        #计算代价函数--第二部分
        error_u = self.R*np.power(x_plus_u[:,3:5],2) # u = x_plus_u[:,3:5],self.R:比例系数（公式中：R）

        # 调试代码
        # print("error_u")
        # print(error_u)

        #将每一项加起来
        sum_u=np.sum(error_u)

        # 调试代码
        # print("sum_u")
        # print(sum_u)

        #计算代价函数总的部分
        J = sum_x+sum_u

        # #调试代码
        # print("J")
        # print(J)

        # time.sleep(1) #休眠比较好调试

        return J 
    

    #------------定义运动学函数------------
    def f(self,x,u):

        f=np.zeros(3) #输入输出是一维的变量，直接定义成数组就行，不要定义成矩阵

        # f[0]= x[0]+ np.cos(u[0])*0.01 # 坐标x
        # f[1]= x[1]+ np.sin(u[0])*0.01 # 坐标y
        # f[2]= x[2]+ np.sin(u[1])*0.01 # 转角theta

        f[0]= x[0]+ np.cos(x[2])*u[0] # 坐标x
        f[1]= x[1]+ np.sin(x[2])*u[0] # 坐标y
        f[2]= x[2]+ u[1] # 转角theta

        return f

    # 1.输入变量符合运动学方程 （等式）
    #def constraint1(self,*args):
    def constraint1(self,*args):
        #数据unpack
        x_plus_u,num = args
        
        #将x_plus_u来reshape成（10,5）矩阵：状态量+控制量
        x_plus_u=np.reshape(x_plus_u,(self.prediction_horizon,self.state_number+self.input_number)) 

        #对u的数据处理
        return self.f(x_plus_u[num,0:3],x_plus_u[num,3:5]) - x_plus_u[num+1,0:3]


    # 2.初始状态达到理想数值(等式）
    def constraint2(self,*args):
        #args的unpack
        x_plus_u,x_init=args

        #将x_plus_u来reshape成（10,5）矩阵：状态量+控制量
        x_plus_u=np.reshape(x_plus_u,(self.prediction_horizon,self.state_number+self.input_number))   

        return x_plus_u[0,:] - x_init # 初始数值要和初始状态相等


    # 3.最终状态达到理想数值(等式）
    def constraint3(self,*args):
        #args的unpack
        x_plus_u,x_d=args

        #将x来reshape成（10,5）矩阵：状态量+控制量
        x_plus_u=np.reshape(x_plus_u,(self.prediction_horizon,self.state_number+self.input_number) )
        
        #对x_d的处理
        x_d = np.array(x_d,dtype=float) # 将元组转化为数组
        x_d = np.reshape(x_d,(self.prediction_horizon,self.state_number) )  # 将数组reshape

        #取数据的size
        lenth = len(x_plus_u[:,0]) # 将prediction horizon取出来
        
        return self.final_error - np.sum(np.fabs(x_plus_u[lenth-1,0:3] - x_d[lenth-1,:]))  # 最终的数值要为零

    # 4.u的转角只能是给定的数值(等式）
    def constraint4(self,*args):
        #数据unpack
        x_plus_u,num = args
        
        #将x来reshape成（10,5）矩阵：状态量+控制量
        x_plus_u=np.reshape(x_plus_u,(self.prediction_horizon,self.state_number+self.input_number) )

        #提取u
        u = x_plus_u[:,3:5]
        
        #reshape u 
        u=np.reshape(u,(1,self.prediction_horizon*self.input_number))

        return u[0,num]-int(u[0,num])


    #------------开始进行优化------------
    
    def optimize(self):
        # 定义空的list 
        cons= []

        # 1.定义运动学约束
        for num in range(self.prediction_horizon-1):
            print("num",num)
            con = {'type': 'eq', 'fun': self.constraint1,'args':(num,)}
            cons.append (con)
        
        # 2.定义初始状态约束
        con10 = {'type': 'eq', 'fun': self.constraint2,'args':(self.x_plus_u_init,)} # 初始状态包括x状态和控制量
        cons.append(con10)

        # # 3.定义最终状态约束
        # con11 = {'type': 'ineq', 'fun': self.constraint3,'args':(self.x_d,)} 
        # cons.append(con11)

        # # 4.定义控制约束
        # for num in range(self.prediction_horizon*self.input_number):
        #     con12 = {'type': 'eq', 'fun': self.constraint4,'args':(num,)} 
        #     cons.append(con12)

        # 总约束：cons
        #localtime_start = time.time()
        starttime = datetime.datetime.now()

        # 求解:1.传入的优化数据是self.x 2.传入的其他数据是args 3.优化数据会传入constrains当中（非常重要）4.传入的参数都被优化了，没传入的参数没有被优化（除了x状态量）
        #solution = minimize(self.objective,self.x,args=(self.x_d,self.u,self.Q,self.R),method='SLSQP',constraints=cons) # 会把初始状态导入数据

        # 优化量：x+u 
        solution = minimize(self.objective,self.x_plus_u,args=(self.x_d,self.Q,self.R),method='SLSQP',constraints=cons) # 会把初始状态导入数据
        x = solution.x 

        #判断是否计算成功
        success = solution.success

        #localtime_end = time.time()
        endtime = datetime.datetime.now()

        # #调试代码:计算时间
        # print("开始时间")
        # print(starttime)

        # print("结束时间")
        # print(endtime)

        # print("运行时间")
        # print(endtime-starttime)

        #将x来reshape成（10,3）矩阵
        x_u_optimize=np.reshape(x,(self.prediction_horizon,self.state_number+self.input_number) )
        
        #将x_u_optimize分解
        x_optimize = x_u_optimize[:,0:3]
        u_optimize = x_u_optimize[:,3:5]

        print("success")
        print(success)

        print("x理想")
        print(self.x_d)

        print("x最终")
        print(x_optimize)

        # print("self.x_init")
        # print(self.x_init)

        print("u最终")
        print(u_optimize)

        #time.sleep(100)
        return x_optimize,u_optimize #将控制数据和状态估计返回

    