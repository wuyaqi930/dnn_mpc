#------------导入官方的package------------
import numpy as np
from scipy.optimize import minimize
import torch 
#------------导入自己的package------------
import mpc 
import plot
import dnn

#------------初始化相关数值------------

#prediction_horizon = 30 #预测范围是10 

##state_number = 3 #状态量有两个
#state_number = 1 #状态量有两个

#input_number = 1 #输入量只有两个，线速度以及角速度

##x_d = 10*np.random.rand( prediction_horizon,state_number ) #初始化理想状态量（列向量）随机数
#sin=np.linspace(2, 5, num=prediction_horizon)
#x_d = 10*np.sin( sin ) #初始化理想状态量（列向量）随机数

##x=np.zeros((prediction_horizon,state_number))#初始化实际状态量（列向量)为零
#x= np.random.rand(prediction_horizon,state_number)#初始化实际状态量（列向量)为随机数

#Q=1000*np.eye(prediction_horizon) #初始化Q（需要半正定）

#R=10*np.eye(prediction_horizon) #初始化R（需要正定）

##u = np.zeros((prediction_horizon,input_number)) #初始化实际状态量（列向量）
#u = np.random.rand(prediction_horizon,input_number) #初始化实际状态量（列向量）为随机数


##------------调用MPC类进行求解------------

##初始化MPC类
#x = mpc.MPC(x,x_d,prediction_horizon,state_number,input_number)

##求解:返回状态X、控制量u
#x,u = x.optimize()


##------------结果绘图------------
##初始化
#polt_it = plot.plot(x,x_d,prediction_horizon,state_number)

##调用绘图函数
#polt_it.draw()

#------------DNN------------
if __name__ == '__main__':

    torch.manual_seed(1)    # reproducible
    
    # 将数据转化成载入并转化成numpy
    data_odom = np.loadtxt('odom.txt',dtype= 'str',skiprows=0,delimiter=",")

    #初始化DNN类
    dnn = dnn.neural_network(data_odom,15000,20)

    #调用优化类 
    dnn.dnn()

    #结果画图
    dnn.plot()




